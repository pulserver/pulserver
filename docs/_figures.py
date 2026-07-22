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


def sequence_figure(sequence, *, time_range, title: str) -> None:
    """Render a fast Pulserver builder through a decoded upstream sequence.

    Its public :meth:`pulserver.pypulseq.Sequence.plot` method delegates to
    the transient upstream sequence exposed as ``sequence._seq``.  Keeping
    the conversion on the public object makes REPL and documentation views
    identical.
    """
    sequence.plot(time_range=time_range, time_disp="ms", plot_now=False)
    plt.gcf().suptitle(title)


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


def _pulse_panels(module, *, with_profile, multidimensional=False):
    """Lay out the RF row, an optional gradient row, and an optional profile.

    The gradient row appears only when the module actually carries gradients,
    so a non-selective pulse still renders as a single envelope panel.
    """
    has_gradients = bool(module_gradients(module)[1])
    if multidimensional:
        width, ratios = 12.5, [1.15, 1.0, 1.0]
    elif with_profile:
        width, ratios = 9.5, [1.15, 1.0]
    else:
        width, ratios = 5.6, [1.0]
    height = 4.2 if has_gradients else 3.3
    rows = [["rf", "mxy", "mz"]] if multidimensional else [["rf", "profile"]] if with_profile else [["rf"]]
    if has_gradients:
        if multidimensional:
            rows.append(["gradient", "mxy", "mz"])
        else:
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

    multidimensional = kind in ("xy", "zf")
    fig, axes, has_gradients = _pulse_panels(
        module,
        with_profile=kind != "none",
        multidimensional=multidimensional,
    )
    _finish_pulse_panels(module, axes, t, b1, title, has_gradients)
    if kind == "none":
        return fig
    profile_axis = None if multidimensional else axes["profile"]

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
        magnetisation = bloch(b1, bz, step)
        profiles = (
            (axes["mxy"], _transverse(magnetisation), "|Mxy|", "viridis", 0.0, 1.0),
            (axes["mz"], magnetisation[:, 2], "Mz", "coolwarm", -1.0, 1.0),
        )
        for profile_axis, values, component, cmap, lower, upper in profiles:
            image = profile_axis.imshow(
                values.reshape(grid_a.shape).T,
                origin="lower",
                aspect=aspect,
                cmap=cmap,
                extent=limits,
                vmin=lower,
                vmax=upper,
            )
            profile_axis.set_xlabel(labels[0])
            profile_axis.set_ylabel(labels[1])
            profile_axis.set_title(f"{component} profile", fontsize=9)
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
        magnetisation = bloch(b1, bz, step)
        axes["profile"].plot(
            axis * scale,
            _transverse(magnetisation),
            lw=1.2,
            color=colour,
            label=f"{label} |Mxy|",
        )
        axes["profile"].plot(
            axis * scale,
            magnetisation[:, 2],
            lw=1.0,
            ls="--",
            color=colour,
            label=f"{label} Mz",
        )
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
    axes["profile"].set_ylabel("M / M0")
    axes["profile"].set_ylim(-1.05, 1.05)
    axes["profile"].set_title("simulated profiles", fontsize=9)
    axes["profile"].legend(fontsize=7)
    return fig


def preparation_figure(module, *, span=1500.0, n_points=201, dt=10e-6, title=None, t2_values=None):
    """Draw a composite preparation, all played gradients, and final Mxy/Mz.

    ``t2_values`` switches the profile panel from off-resonance to the ideal
    stored T2-weighted longitudinal response of a :class:`T2PrepPulse`.
    """
    import pypulseq as pp

    blocks = tuple(module)
    durations = np.asarray([pp.calc_duration(*block) for block in blocks])
    starts = np.concatenate(([0.0], np.cumsum(durations[:-1])))
    total = float(durations.sum())
    time = (np.arange(max(1, int(np.ceil(total / dt)))) + 0.5) * dt
    b1 = np.zeros(time.size, dtype=complex)
    gradient_traces = {}
    for block_start, block in zip(starts, blocks, strict=True):
        for event in block:
            if getattr(event, "type", None) == "rf":
                event_time = block_start + float(event.delay) + np.asarray(event.t)
                signal = np.asarray(event.signal) * np.exp(
                    1j * (2.0 * np.pi * float(event.freq_offset) * event_time + float(event.phase_offset))
                )
                active = (time >= event_time[0]) & (time <= event_time[-1])
                b1[active] += np.interp(time[active], event_time, signal.real) + 1j * np.interp(
                    time[active], event_time, signal.imag
                )
            elif getattr(event, "type", None) in ("grad", "trap"):
                channel = event.channel
                gradient_traces.setdefault(channel, np.zeros(time.size))
                gradient_traces[channel] += sample_gradient(event, time - block_start)

    if t2_values is None:
        offsets = np.linspace(-span, span, n_points)
        magnetisation = bloch(b1, offsets[:, None], dt)
    if gradient_traces:
        fig, (waveform, gradient, profile) = plt.subplots(1, 3, figsize=(13.0, 3.4), layout="constrained")
    else:
        fig, (waveform, profile) = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")
        gradient = None
    waveform.plot(time * 1e3, b1.real, lw=1.0, label="real")
    waveform.plot(time * 1e3, b1.imag, lw=1.0, label="imag")
    waveform.set(xlabel="t [ms]", ylabel="B1 [Hz]", title=title or type(module).__name__)
    waveform.legend(fontsize=7)
    if gradient is not None:
        for channel, trace in gradient_traces.items():
            gradient.plot(time * 1e3, trace * 1e-3, lw=1.0, color=_GRADIENT_COLOURS[channel], label=f"G{channel}")
        gradient.axhline(0.0, lw=0.6, color="0.7")
        gradient.set(xlabel="t [ms]", ylabel="G [kHz/m]", title="played gradients")
        gradient.legend(fontsize=7)
    if t2_values is None:
        profile.plot(offsets, _transverse(magnetisation), label="|Mxy|")
        profile.plot(offsets, magnetisation[:, 2], ls="--", label="Mz")
        profile.set(xlabel="off-resonance [Hz]", title="final magnetization")
    else:
        t2_values = np.asarray(t2_values, dtype=float)
        if t2_values.ndim != 1 or t2_values.size < 2 or np.any(t2_values <= 0):
            raise ValueError("t2_values must be a one-dimensional array of at least two positive values")
        echo_time = float(module.echo_time)
        direction = -1.0 if getattr(module, "final_tip", "up") == "down" else 1.0
        profile.plot(t2_values * 1e3, np.zeros_like(t2_values), label="|Mxy|")
        profile.plot(t2_values * 1e3, direction * np.exp(-echo_time / t2_values), ls="--", label="Mz")
        flip = "180°" if direction < 0 else "0°"
        profile.set(
            xlabel="assumed T2 [ms]",
            title=f"T2-weighted stored M (effective {flip})",
        )
    profile.set(ylabel="M / M0", ylim=(-1.05, 1.05))
    profile.legend(fontsize=7)
    return fig


def frequency_profile_gallery(entries, *, extent=2800.0, n_points=401, title=None):
    """Compare the Mxy and Mz frequency profiles of several RF modules."""
    offsets = np.linspace(-extent, extent, n_points)
    fig, axes = plt.subplots(1, len(entries), figsize=(4.0 * len(entries), 3.4), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for axis, (label, module) in zip(axes, entries, strict=True):
        rf = getattr(module, "rf", module)
        time, b1 = _rf_grid(rf)
        magnetisation = bloch(b1, offsets[:, None], float(time[1] - time[0]))
        axis.plot(offsets, _transverse(magnetisation), label="|Mxy|")
        axis.plot(offsets, magnetisation[:, 2], ls="--", label="Mz")
        axis.set(xlabel="off-resonance [Hz]", title=label, ylim=(-1.05, 1.05))
        axis.legend(fontsize=7)
    axes[0].set_ylabel("M / M0")
    if title:
        fig.suptitle(title)
    return fig


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------


def _readout_trajectory(module):
    readout = getattr(module, "readout", module)
    trajectory = np.asarray(readout.trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError("readout trajectory must have at least two columns")
    return trajectory


def _square_trajectory_axes(axis, trajectory):
    lower = np.min(trajectory[:, :2], axis=0)
    upper = np.max(trajectory[:, :2], axis=0)
    center = 0.5 * (lower + upper)
    half_width = max(1.0, 0.55 * float(np.max(upper - lower)))
    axis.set_xlim(center[0] - half_width, center[0] + half_width)
    axis.set_ylim(center[1] - half_width, center[1] + half_width)
    axis.set_aspect("equal", adjustable="box")


def trajectory_figure(entries, *, title=None):
    """Show canonical in-plane kx'/ky' paths for one or more readouts."""
    fig, axes = plt.subplots(
        1,
        len(entries),
        figsize=(3.4 * len(entries), 3.4),
        squeeze=False,
        layout="constrained",
    )
    for axis, (label, module) in zip(axes[0], entries, strict=True):
        trajectory = _readout_trajectory(module)
        axis.plot(trajectory[:, 0], trajectory[:, 1], lw=0.9)
        axis.scatter(trajectory[0, 0], trajectory[0, 1], s=18, marker="o", label="start")
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], s=22, marker="x", label="end")
        axis.set(xlabel="kx' [cycles/m]", ylabel="ky' [cycles/m]", title=label)
        _square_trajectory_axes(axis, trajectory)
        axis.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    return fig


def noncartesian_readout_figure(module, *, title=None):
    """Show the played blocks and canonical in-plane trajectory together."""
    plot = module.plot()
    waveform_figure = plot.fig1
    if title:
        waveform_figure.suptitle(title)
    waveform_figure.canvas.draw()
    waveform_pixels = np.asarray(waveform_figure.canvas.buffer_rgba()).copy()
    plt.close(waveform_figure)

    trajectory = _readout_trajectory(module)
    fig, (waveform, path) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": (2.3, 1.0)},
        layout="constrained",
    )
    waveform.imshow(waveform_pixels)
    waveform.axis("off")
    path.plot(trajectory[:, 0], trajectory[:, 1], lw=0.9)
    path.scatter(trajectory[0, 0], trajectory[0, 1], s=18, marker="o", label="start")
    path.scatter(trajectory[-1, 0], trajectory[-1, 1], s=22, marker="x", label="end")
    path.set(
        xlabel="kx' [cycles/m]",
        ylabel="ky' [cycles/m]",
        title=f"{title or type(module).__name__}: canonical trajectory",
    )
    _square_trajectory_axes(path, trajectory)
    path.legend(fontsize=7)
    return fig


def zte_readout_figure(module, *, title="ZTE"):
    """Show a ZTE segment and its sampled in-plane half-spokes."""
    plot = module.plot()
    waveform_figure = plot.fig1
    waveform_figure.suptitle(title)
    waveform_figure.canvas.draw()
    waveform_pixels = np.asarray(waveform_figure.canvas.buffer_rgba()).copy()
    plt.close(waveform_figure)
    start = (module.num_missing_samples + 0.5) * module.delta_k
    stop = module.kmax
    radii = np.linspace(start, stop, module.n_samples)
    fig, (waveform, axis) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": (2.3, 1.0)},
        layout="constrained",
    )
    waveform.imshow(waveform_pixels)
    waveform.axis("off")
    for direction in module.directions:
        axis.plot(radii * direction[0], radii * direction[1], lw=0.8)
    axis.set(
        xlabel="kx' [cycles/m]",
        ylabel="ky' [cycles/m]",
        title=f"{title}: acquired trajectory",
    )
    axis.set_aspect("equal", adjustable="box")
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


def _blip_parabola(xy1, xy2, *, num=11):
    """Points tracing the two back-to-back parabolic arcs of a gradient blip.

    A trapezoidal z-blip's k-space trajectory (its time integral) is a
    parabola on the way up and another on the way down; connecting two
    samples this way rather than with a straight line is what makes a
    skipped-CAIPI trajectory figure read as a *played* gradient waveform
    instead of an abstract point-to-point path — see Stirnberg & Stöcker,
    MRM 2020, DOI ``10.1002/mrm.28486``.
    """
    x1, y1 = xy1
    x2, y2 = xy2
    x = np.linspace(x1, x2, num)
    half = (num + 1) // 2
    a = 2.0 * (y2 - y1) / (x2 - x1) ** 2 if x2 != x1 else 0.0
    y = np.empty_like(x)
    y[:half] = a * (x[:half] - x1) ** 2 + y1
    y[half:] = -a * (x[half:] - x2) ** 2 + y2
    return x, y


def epi_sampling_figure(plans, *, cmap="tab10"):
    """Show one or more EPI/skipped-CAIPI plans as shot trajectories on their lattice.

    Every sampled location of the plan's full Cartesian lattice is drawn as a
    bordered checkerboard cell, then every shot's phase-encode trajectory is
    overlaid as blip-parabola-connected points: shot 0 bold in front, the
    remaining shots faint behind it. This is the same visual grammar as
    Stirnberg & Stöcker's skipped-CAIPI reference figures (MRM 2020, DOI
    ``10.1002/mrm.28486``) — the segmented interleaves read off directly as
    parallel paths through one shared lattice. A 2D (ky-only) plan has no
    lattice to grid against, so each shot is instead drawn on its own row.

    Axes carry their ky/kz labels; panels are left untitled, so the
    surrounding prose names them.
    """
    accent = plt.get_cmap(cmap)(3)
    fig, axes = plt.subplots(1, len(plans), figsize=(4.6 * len(plans), 3.7), squeeze=False)
    for axis, plan in zip(axes[0], plans, strict=True):
        # Accept either an AcquisitionPlan or a bare SamplingPattern.
        sampling = getattr(plan, "sampling", plan)
        n_shots = sampling.n_shots
        if sampling.support.shape[1] == 2 and sampling.mask is not None:
            ny, nz = sampling.mask.shape
            # Reference convention: unsampled cells mid-grey, sampled white.
            axis.pcolor(
                sampling.mask.T.astype(float) + 0.3,
                cmap="gray",
                edgecolors="0.2",
                linewidth=0.5,
                vmin=0.0,
                vmax=1.0,
            )
            for shot in reversed(range(n_shots)):
                c = sampling[shot] + 0.5
                bold = shot == 0
                # Reference convention: the leading shot in colour, the
                # remaining interleaves darkened behind it.
                colour = accent if bold else tuple(0.45 * channel for channel in accent[:3])
                for (x1, y1), (x2, y2) in zip(c[:-1], c[1:], strict=True):
                    x, y = _blip_parabola((x1, y1), (x2, y2))
                    axis.plot(x, y, color=colour, lw=1.4 if bold else 0.6, zorder=3 if bold else 2)
                axis.scatter(
                    c[:, 0], c[:, 1], color=colour, s=14 if bold else 5, zorder=4 if bold else 2
                )
            axis.set(xlabel="ky", ylabel="kz", aspect="equal", xlim=(0, ny), ylim=(0, nz))
        else:
            colours = plt.get_cmap(cmap)
            for shot in range(n_shots):
                c = sampling[shot]
                axis.plot(
                    c[:, 0], np.full(len(c), shot), marker="o", ms=3.5, lw=1.0, color=colours(shot % 10)
                )
            axis.set(xlabel="ky", ylabel="shot")
            axis.set_yticks(range(n_shots))
    fig.tight_layout()
    return fig
