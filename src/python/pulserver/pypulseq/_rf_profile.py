"""What a pulse does to the magnetisation, drawn.

:func:`pulserver.pypulseq.sim_rf` answers for one pulse across off-resonance,
and :func:`pulserver.pypulseq.bloch` for anything else. Neither says what to
draw. This is the picture: the envelope beside the profile it produces, over
one axis or over a plane, integrated over everything that is played when a
single pulse is not the whole story.

Reached through :meth:`pulserver.pypulseq.Sequence.plot_rf` and
:meth:`pulserver.design.RfModule.plot_rf`. Either way what arrives here is a
*source*: anything that can list its blocks and report its system limits.
"""

from __future__ import annotations

__all__ = ["plot_rf"]

import numpy as np

from . import _style
from ._common import _rf_use

#: What each ``use`` asks of a pulse, and so which response answers for it.
#: The key names the :class:`~pulserver.pypulseq.RfResponse` field, the label
#: is what the profile axis is called, and the index selects a hue.
_RESPONSE = {
    "excitation": ("mz_xy", "$|M_{xy}|$", 0),
    "refocusing": ("ref_eff", "refocusing efficiency", 2),
    "inversion": ("mz_z", "$M_z$", 1),
    "saturation": ("mz_z", "$M_z$", 1),
    "preparation": ("mz_z", "$M_z$", 1),
    "other": ("mz_xy", "$|M_{xy}|$", 0),
}

#: What each letter of a ``plane`` draws against: the gradient channel that
#: makes the pulse selective along it, and the axis label. ``f`` is
#: off-resonance, which no gradient modulates.
_AXES = {
    "x": (0, "x [mm]"),
    "y": (1, "y [mm]"),
    "z": (2, "z [mm]"),
    "f": (None, "off-resonance [Hz]"),
}


def _blocks_of(source) -> list[tuple]:
    """One tuple of events per block, from a module or from a sequence.

    A :class:`~pulserver.SequenceModule` publishes exactly this; a
    :class:`~pulserver.pypulseq.Sequence` stores blocks and hands them back as
    named fields, so those are flattened to the same shape.
    """
    blocks = getattr(source, "blocks", None)
    if blocks is not None:
        return list(blocks)
    from . import block_to_events

    return [
        tuple(block_to_events(source.get_block(number)))
        for number in range(1, source.num_blocks + 1)
    ]


def _system_of(source):
    """The system limits a source was built under."""
    system = getattr(source, "system", None)
    return source.seq.system if system is None else system


def _first_rf(blocks, use: str | None = None):
    """The first RF event played, by type rather than by name.

    ``use`` narrows it to a pulse tagged for that job, which is how one pulse
    of a full sequence is named without holding a reference to the event: a
    spin echo's refocusing pulse is ``"refocusing"`` wherever in the
    repetition it falls.
    """
    for block in blocks:
        for event in block:
            if getattr(event, "type", None) != "rf":
                continue
            if use is None or _rf_use(event) == use:
                return event
    played = "RF pulse" if use is None else f"pulse used for {use!r}"
    raise ValueError(f"nothing here plays a {played}")


def _gradient_at(event, time: float) -> float:
    """The gradient's amplitude at ``time``, in Hz/m, on the block's clock."""
    delay = float(event.delay)
    if event.type == "trap":
        rise, flat, fall = (
            float(event.rise_time),
            float(event.flat_time),
            float(event.fall_time),
        )
        times = [delay, delay + rise, delay + rise + flat, delay + rise + flat + fall]
        amplitudes = [0.0, float(event.amplitude), float(event.amplitude), 0.0]
    else:
        times = delay + np.asarray(event.tt, dtype=float)
        amplitudes = np.asarray(event.waveform, dtype=float)
    return float(np.interp(time, times, amplitudes, left=0.0, right=0.0))


def _selection(blocks, pulse) -> tuple[int | None, float]:
    """Which channel the pulse is selective along, and at what amplitude.

    The gradient that makes a pulse spatially selective is the one played in
    its own block, so it is read from there rather than from a name: a readout
    calls it ``gz_ref`` where an excitation calls it ``gz``. It is sampled at
    the pulse's own centre, which is the only amplitude the profile is
    selective under -- a refocusing lobe with crushers bridged onto it reaches
    several others either side of the pulse.
    """
    centre = float(pulse.delay) + float(pulse.center)
    for block in blocks:
        if not any(event is pulse for event in block):
            continue
        for event in block:
            if getattr(event, "type", None) in ("trap", "grad"):
                amplitude = _gradient_at(event, centre)
                if abs(amplitude) > 1.0:
                    return "xyz".index(event.channel), amplitude
    return None, 0.0


def _on_one_raster(source, dt: float):
    """The source's RF and gradients, resampled onto one uniform raster.

    ``waveforms_and_times`` reports each channel as its own ``(time, value)``
    pair on the breakpoints it needs; a Bloch integration wants all four on
    the same steps.
    """
    channels = source.waveforms_and_times(True, compat=False).waveforms
    parts = [channels.gx, channels.gy, channels.gz, channels.rf]
    parts = [np.atleast_2d(np.asarray(p)) if p is not None else None for p in parts]
    stop = max(float(p[0, -1].real) for p in parts if p is not None and p.size)
    times = np.arange(0.5 * dt, stop, dt)

    def resample(part, complex_values: bool):
        empty = np.zeros(times.size, dtype=complex if complex_values else float)
        if part is None or part.size == 0:
            return empty
        clock, values = part[0].real, part[1]
        sampled = np.interp(times, clock, values.real, left=0.0, right=0.0)
        if complex_values:
            sampled = sampled + 1j * np.interp(
                times, clock, values.imag, left=0.0, right=0.0
            )
        return sampled

    gradients = [resample(part, False) for part in parts[:3]]
    return times, resample(parts[3], True), gradients


def _field(axis_values, channel, gradients, times):
    """The longitudinal field each grid point sees, in Hz, over ``times``."""
    if channel is None:
        return np.asarray(axis_values, dtype=float)[:, None] * np.ones_like(times)
    return np.outer(1e-3 * np.asarray(axis_values, dtype=float), gradients[channel])


def _responses(b1, field, dt: float) -> dict:
    """Every response a ``use`` can ask for, from three starting states."""
    from . import bloch

    from_z, from_x, from_y = (
        bloch(b1, field, dt, initial=start)
        for start in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    return {
        "mz_z": from_z[:, 2],
        "mz_xy": from_z[:, 0] + 1j * from_z[:, 1],
        "ref_eff": (
            (from_x[:, 0] + 1j * from_x[:, 1]) + 1j * (from_y[:, 0] + 1j * from_y[:, 1])
        )
        / 2.0,
    }


def _limits(extent, widen: float | None = None):
    """``extent`` as a ``(low, high)`` pair, optionally widened."""
    if extent is None:
        return None if widen is None else (-widen, widen)
    if np.ndim(extent) == 0:
        low, high = -float(extent), float(extent)
    else:
        low, high = float(extent[0]), float(extent[1])
    if widen is None:
        return low, high
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    return centre - 2.0 * half, centre + 2.0 * half


def _default_span(letter, bandwidth: float, amplitude: float, centre: float):
    """How wide to simulate along ``letter`` when nothing was asked for."""
    if letter == "f":
        # Wide enough to hold more than the passband: what a spectral axis is
        # drawn for is where the response comes back, and a subpulse train
        # repeats well outside its own bandwidth.
        return -4.0 * bandwidth, 4.0 * bandwidth
    if abs(amplitude) > 1.0:
        width = 1e3 * (abs(bandwidth) + abs(centre)) / abs(amplitude)
        return -width, width
    return -20.0, 20.0


def plot_rf(
    source,
    pulse=None,
    *,
    plane: str | None = None,
    kind: str | None = None,
    extent=None,
    span=None,
    samples: int = 401,
    dt: float = 8e-6,
    whole: bool = False,
    title: str | None = None,
    plot_now: bool = True,
):
    """Draw ``source``'s pulse beside the magnetisation profile it produces.

    See :meth:`pulserver.pypulseq.Sequence.plot_rf`, which is how this is
    reached and where the arguments are documented.
    """
    from matplotlib import pyplot as plt

    from . import calc_rf_bandwidth, sim_rf

    blocks = _blocks_of(source)
    if pulse is None or isinstance(pulse, str):
        pulse = _first_rf(blocks, pulse)
    channel, amplitude = _selection(blocks, pulse)
    if plane is None:
        plane = "xyz"[channel] if channel is not None else "f"
    if len(plane) not in (1, 2) or any(letter not in _AXES for letter in plane):
        raise ValueError(
            f"plot_rf(): plane must be one or two of {', '.join(_AXES)}, got {plane!r}"
        )

    field_name, profile_label, hue = _RESPONSE.get(
        kind or _rf_use(pulse), _RESPONSE["other"]
    )
    bandwidth = float(calc_rf_bandwidth(pulse))
    centre = float(getattr(pulse, "freq_offset", 0.0))

    if whole or len(plane) == 2:
        times, b1, gradients = _on_one_raster(source, dt)
    else:
        # One pulse across off-resonance is what ``sim_rf`` already answers,
        # on the pulse's own raster and without the crushers around it.
        times, b1, gradients = None, None, None

    grids, limits = [], []
    for letter, asked in zip(plane, (extent, span), strict=False):
        low, high = _limits(asked) or _default_span(
            letter, bandwidth, amplitude, centre
        )
        points = samples if len(plane) == 1 else min(samples, 91)
        grids.append(np.linspace(low, high, points))
        limits.append((low, high))

    system = _system_of(source)
    if len(plane) == 1 and not whole:
        simulated = sim_rf(pulse, compat=False)
        values = np.asarray(getattr(simulated, field_name))
        axis_values = np.asarray(simulated.frequency, dtype=float)
        if plane != "f":
            axis_values = 1e3 * axis_values / amplitude
        clock = 1e3 * np.asarray(pulse.t, dtype=float)
        envelope = 1e6 * np.abs(np.asarray(pulse.signal)) / float(system.gamma)
    else:
        clock = 1e3 * times
        envelope = 1e6 * np.abs(b1) / float(system.gamma)
        if len(plane) == 1:
            axis_values = grids[0]
            field = _field(axis_values, _AXES[plane][0], gradients, times)
            values = _responses(b1, field, dt)[field_name]
        else:
            first, second = np.meshgrid(grids[0], grids[1], indexing="ij")
            field = _field(
                first.ravel(), _AXES[plane[0]][0], gradients, times
            ) + _field(second.ravel(), _AXES[plane[1]][0], gradients, times)
            responses = _responses(b1, field, dt)
            values = None

    figure = _draw(
        plane=plane,
        clock=clock,
        envelope=envelope,
        values=values,
        responses=None if len(plane) == 1 else responses,
        shape=None if len(plane) == 1 else first.shape,
        axis_values=None if len(plane) == 2 else axis_values,
        limits=limits,
        profile_label=profile_label,
        hue=hue,
        title=title,
    )
    if plot_now:
        plt.show()
    return figure


def _envelope_panel(axis, clock, envelope) -> None:
    axis.plot(clock, envelope, color=_style.SERIES[0], lw=1.6)
    axis.fill_between(clock, envelope, color=_style.SERIES[0], alpha=0.12, lw=0)
    axis.set_xlabel("time [ms]")
    axis.set_ylabel(r"$|B_1|$ [$\mu$T]")
    axis.set_xlim(clock[0], clock[-1])
    _style.axis_style(axis, "envelope")


def _draw(
    *,
    plane,
    clock,
    envelope,
    values,
    responses,
    shape,
    axis_values,
    limits,
    profile_label,
    hue,
    title,
):
    """The figure itself: the envelope, then one profile axis or two heatmaps."""
    from matplotlib import pyplot as plt

    if len(plane) == 1:
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": (1.0, 1.35)}
        )
        _envelope_panel(left, clock, envelope)

        drawn = np.abs(values) if np.iscomplexobj(values) else np.asarray(values)
        order = np.argsort(axis_values)
        right.axhline(0.0, color=_style.FAINT, lw=0.8)
        right.plot(
            np.asarray(axis_values)[order],
            drawn[order],
            color=_style.SERIES[hue],
            lw=1.8,
        )
        right.set_xlabel(_AXES[plane][1])
        right.set_ylabel(profile_label)
        right.set_xlim(*limits[0])
        _style.axis_style(right, "profile")
    else:
        figure, (envelope_axis, mxy, mz) = plt.subplots(
            1, 3, figsize=(11.2, 3.1), gridspec_kw={"width_ratios": (1.0, 1.1, 1.1)}
        )
        _envelope_panel(envelope_axis, clock, envelope)

        transverse = np.abs(responses["mz_xy"]).reshape(shape)
        longitudinal = np.real(responses["mz_z"]).reshape(shape)
        spatial = "f" not in plane
        panels = (
            (mxy, transverse, r"$|M_{xy}|$", _style.MAGNITUDE, 0.0, 1.0),
            (mz, longitudinal, "$M_z$", _style.SIGNED, -1.0, 1.0),
        )
        for axis, grid, label, cmap, low, high in panels:
            image = axis.imshow(
                grid.T,
                origin="lower",
                aspect="equal" if spatial else "auto",
                cmap=cmap,
                vmin=low,
                vmax=high,
                interpolation="nearest",
                extent=(*limits[0], *limits[1]),
            )
            axis.set_xlabel(_AXES[plane[0]][1])
            axis.set_ylabel(_AXES[plane[1]][1])
            bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors=_style.MUTED, labelsize=8, length=0)
            _style.image_style(axis, label)

    _style.figure_title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.92 if title else 1.0))
    return figure
