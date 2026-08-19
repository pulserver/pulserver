"""Figures the API docstrings draw.

Documentation-only. The ``.. plot::`` directives embedded in Pulserver's
docstrings import this package; nothing in the shipped wheel does, which is
why it lives beside ``conf.py`` rather than under ``src/``.

Four kinds of picture, one function each:

``excitation_kspace``
    The path a multidimensional pulse deposits its energy along.
``rf_profile``
    What a pulse does to the magnetisation, beside the envelope that does it.
    The Bloch integration is :func:`pulserver.pypulseq.sim_rf`, reached
    through :meth:`pulserver.design.RfModule.sim_rf`.
``trajectory``
    Where a readout's ADC samples land in k-space, coloured by the echo or
    the shot that acquired them, so an echo train reads as an ordering rather
    than as one shape.
``order_figure``
    The ``(ky, kz)`` views an ordering deals into its trains, coloured by the
    echo they are encoded at.

Every function returns the :class:`~matplotlib.figure.Figure` it drew, so a
directive that wants to add to it can.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

__all__ = [
    "ORDINAL",
    "SERIES",
    "excitation_kspace",
    "order_figure",
    "rf_profile",
    "trajectory",
]

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------

#: Ink, in decreasing prominence, and the hairline everything recessive is
#: drawn in.
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#b9b8b2"

#: Categorical hues, assigned in this order and never cycled. Identity, not
#: magnitude: a gradient axis, a pulse against its profile.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

#: One hue, light to dark, for an ordered quantity — an echo index, a shot
#: number. The lightest step still reads against white paper.
ORDINAL = LinearSegmentedColormap.from_list(
    "pulserver-ordinal",
    ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"],
)


def _style(axis, title: str = "") -> None:
    """Recessive frame: two spines, ticks that do not shout, no grid."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(FAINT)
    axis.grid(False)
    axis.set_facecolor("none")
    axis.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    if title:
        axis.set_title(title, loc="left", fontsize=9, color=INK)


def _title(figure, text: str | None) -> None:
    if text:
        figure.suptitle(text, x=0.01, ha="left", fontsize=10, color=INK)


def _colorbar(figure, axis, values, label: str, pad: float = 0.03):
    """A discrete ordinal bar, ticked at every step while they are few."""
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=ORDINAL), ax=axis, fraction=0.045, pad=pad
    )
    bar.outline.set_visible(False)
    bar.set_label(label, color=MUTED, fontsize=8)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    span = norm.vmax - norm.vmin
    if not span:
        bar.set_ticks([norm.vmin])
    elif span <= 8:
        bar.set_ticks(np.arange(norm.vmin, norm.vmax + 1))
    else:
        bar.set_ticks([norm.vmin, norm.vmax])
    return bar


# ----------------------------------------------------------------------
# RF: envelope and profile
# ----------------------------------------------------------------------

#: What each ``use`` asks of a pulse, and so which response answers for it.
#: ``key`` names the :class:`~pulserver.pypulseq.RfResponse` field, and the
#: label is what the profile axis is called.
_RESPONSE = {
    "excitation": ("mz_xy", "$|M_{xy}|$", 0),
    "refocusing": ("ref_eff", "refocusing efficiency", 2),
    "inversion": ("mz_z", "$M_z$", 1),
    "saturation": ("mz_z", "$M_z$", 1),
    "preparation": ("mz_z", "$M_z$", 1),
    "other": ("mz_xy", "$|M_{xy}|$", 0),
}


def _system(module):
    """The limits the module was built against, from the sequence it built."""
    return module.seq.system


def _first_rf(module):
    """The first RF event a module plays, by type rather than by name."""
    for block in module.blocks:
        for event in block:
            if getattr(event, "type", None) == "rf":
                return event
    raise ValueError(f"{type(module).__name__} plays no RF pulse")


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


def _selection_amplitude(module, pulse) -> float:
    """Hz/m the pulse is selective under, or zero when it is not.

    The gradient that makes a pulse spatially selective is the one played in
    its own block, so it is read from there rather than from a name: a
    readout calls it ``gz_ref`` where an excitation calls it ``gz``. It is
    sampled at the pulse's own centre, which is the only amplitude the
    profile is selective under -- a refocusing lobe with crushers bridged
    onto it reaches several others either side of the pulse.
    """
    centre = float(pulse.delay) + float(pulse.center)
    for block in module.blocks:
        if pulse not in block:
            continue
        for event in block:
            if getattr(event, "type", None) in ("trap", "grad"):
                return _gradient_at(event, centre)
    return 0.0


def _on_one_raster(module, dt: float):
    """The module's RF and gradients, resampled onto one uniform raster.

    ``waveforms_and_times`` reports each channel as its own ``(time, value)``
    pair on the breakpoints it needs; a Bloch integration wants all four on
    the same steps.
    """
    channels = module.waveforms_and_times(True, compat=False).waveforms
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


def _whole_module_response(module, axis_values, spatial: bool, axis: int, dt: float):
    """Integrate the Bloch equation over everything the module plays.

    :func:`pulserver.pypulseq.sim_rf` answers for one pulse. A preparation is
    several, with the crushers and the free precession between them, and only
    the whole of it has a profile worth drawing.
    """
    from pulserver import pypulseq as pp

    times, b1, gradients = _on_one_raster(module, dt)
    if spatial:
        field = np.outer(1e-3 * axis_values, gradients[axis])
    else:
        field = np.asarray(axis_values, dtype=float)[:, None] * np.ones_like(times)

    from_z, from_x, from_y = (
        pp.bloch(b1, field, dt, initial=start)
        for start in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    return {
        "mz_z": from_z[:, 2],
        "mz_xy": from_z[:, 0] + 1j * from_z[:, 1],
        "ref_eff": (
            (from_x[:, 0] + 1j * from_x[:, 1]) + 1j * (from_y[:, 0] + 1j * from_y[:, 1])
        )
        / 2.0,
    }, (times, b1)


def rf_profile(
    module,
    *,
    kind: str | None = None,
    title: str | None = None,
    extent=None,
    spatial: bool | None = None,
    whole: bool = False,
    span: float | None = None,
    samples: int = 401,
    dt: float = 4e-6,
    figsize: tuple[float, float] = (8.4, 2.9),
    **simulation,
):
    """Draw a pulse beside the magnetisation profile it produces.

    Parameters
    ----------
    module : pulserver.design.RfModule
        The module whose pulse to simulate.
    kind : str, optional
        Which response to draw: ``"excitation"``, ``"refocusing"``,
        ``"inversion"`` or ``"saturation"``. Read off the pulse's ``use`` by
        default.
    title : str, optional
        Figure title.
    extent : float or tuple of float, optional
        The profile axis, in millimetres when the pulse is spatially
        selective and in hertz when it is not: a half-width about zero, or an
        explicit ``(low, high)``. The whole simulated axis by default.
    spatial : bool, optional
        Plot against position rather than frequency. By default the pulse
        decides: one played under a gradient is spatially selective, one
        played without is not.
    whole : bool, optional
        Integrate everything the module plays — every pulse, every crusher,
        and the free precession between them — rather than its first pulse
        alone. What a preparation module's profile means.
    span : float, optional
        Half-width of the simulated axis under ``whole``, in the units of
        ``extent``. Twice ``extent`` by default.
    samples : int, optional
        Points on the simulated axis under ``whole``.
    dt : float, optional
        Integration raster under ``whole``, in seconds.
    figsize : tuple of float, optional
        Figure size, in inches.
    **simulation
        Forwarded to :meth:`pulserver.design.RfModule.sim_rf`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pulse = _first_rf(module)
    use = str(getattr(pulse, "use", "") or "other")
    field, profile_label, colour = _RESPONSE.get(kind or use, _RESPONSE["other"])
    amplitude = _selection_amplitude(module, pulse)
    if spatial is None:
        spatial = abs(amplitude) > 1.0

    if whole:
        if span is None and extent is None:
            # Nothing was asked for, so simulate what the pulse can reach: a
            # couple of its own bandwidths either side of where it sits.
            from pulserver import pypulseq as pp

            width = float(pp.calc_rf_bandwidth(pulse))
            centre = float(getattr(pulse, "freq_offset", 0.0))
            span = abs(width) + abs(centre)
            if spatial:
                span *= 1e3 / abs(amplitude)
        limits = _limits(extent, span)
        axis_values = np.linspace(limits[0], limits[1], samples)
        response, (clock, b1) = _whole_module_response(
            module, axis_values, spatial, _selection_axis(module, pulse), dt
        )
        values = response[field]
        envelope_t = 1e3 * clock
        envelope = 1e6 * np.abs(b1) / float(_system(module).gamma)
    else:
        simulated = module.sim_rf(pulse, compat=False, **simulation)
        values = np.asarray(getattr(simulated, field))
        axis_values = simulated.frequency
        if spatial:
            axis_values = 1e3 * axis_values / amplitude
        envelope_t = 1e3 * np.asarray(pulse.t, dtype=float)
        envelope = 1e6 * np.abs(np.asarray(pulse.signal)) / float(_system(module).gamma)

    if np.iscomplexobj(values):
        values = np.abs(values)
    order = np.argsort(axis_values)
    axis_values, values = np.asarray(axis_values)[order], np.asarray(values)[order]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": (1.0, 1.35)}
    )

    left.plot(envelope_t, envelope, color=SERIES[0], lw=1.6)
    left.fill_between(envelope_t, envelope, color=SERIES[0], alpha=0.12, lw=0)
    left.set_xlabel("time [ms]")
    left.set_ylabel(r"$|B_1|$ [$\mu$T]")
    left.set_xlim(envelope_t[0], envelope_t[-1])
    _style(left, "envelope")

    right.axhline(0.0, color=FAINT, lw=0.8)
    right.plot(axis_values, values, color=SERIES[colour], lw=1.8)
    right.set_xlabel("position [mm]" if spatial else "off-resonance [Hz]")
    right.set_ylabel(profile_label)
    right.set_xlim(*(_limits(extent) or (axis_values[0], axis_values[-1])))
    _style(right, "profile")

    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    return figure


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


def _selection_axis(module, pulse) -> int:
    """Which gradient channel the pulse is selective along."""
    for block in module.blocks:
        if pulse not in block:
            continue
        for event in block:
            if getattr(event, "type", None) in ("trap", "grad"):
                return "xyz".index(event.channel)
    return 2


# ----------------------------------------------------------------------
# k-space
# ----------------------------------------------------------------------


def _scaled(event, factor, cache):
    """``event`` at ``factor`` of its amplitude, built once per factor."""
    from pulserver import pypulseq as pp

    key = (id(event), round(float(factor), 12))
    if key not in cache:
        cache[key] = pp.scale_grad(event, float(factor))
    return cache[key]


def _replay(module, ky, kz, per: str):
    """Play a readout module's blocks as a scan loop would.

    The module publishes its phase encodes at full amplitude for the loop to
    scale, so the trajectory of its own sequence is one line repeated. This
    plays the same blocks with one scale factor per echo (``per="echo"``, a
    train) or one per pass through the whole module (``per="shot"``, a
    Cartesian TR played several times), which is the picture a loop produces.
    """
    from pulserver import pypulseq as pp

    events = module.events
    # ``gz_pre`` is a partition encode in a 3D readout and a slice rephaser in
    # a 2D one, and only the caller knows which module this is: the z axis is
    # scaled exactly when partition steps were asked for.
    axes = [(0, ("gy_pre", "gy_rew"))]
    if kz is not None:
        axes.append((1, ("gz_partition", "gz_partition_rew", "gz_pre", "gz_rew")))
    encodes = {
        getattr(events, name, None): (axis, name.endswith("rew"))
        for axis, names in axes
        for name in names
    }
    encodes.pop(None, None)
    scales = (
        np.atleast_1d(np.asarray(ky, dtype=float)),
        np.atleast_1d(np.asarray(ky if kz is None else kz, dtype=float)),
    )
    passes = 1 if per == "echo" else len(scales[0])

    sequence = pp.Sequence(_system(module))
    cache: dict = {}
    for shot in range(passes):
        acquired = 0
        for block in module.blocks:
            has_adc = any(getattr(e, "type", None) == "adc" for e in block)
            played = []
            for event in block:
                axis_rewinder = encodes.get(event)
                if axis_rewinder is None:
                    played.append(event)
                    continue
                axis, is_rewinder = axis_rewinder
                step = shot if per == "shot" else acquired - int(is_rewinder)
                factor = scales[axis][min(step, len(scales[axis]) - 1)]
                played.append(_scaled(event, factor, cache))
            sequence.add_block(*played)
            if has_adc:
                acquired += 1
    return sequence


def _rotations(angles, axis: str):
    """``angles`` as rotations, in whichever of three ways it is written.

    A turn about ``axis`` per entry, a stack of rotation matrices, or the
    directions a projection acquisition covers -- the last turned into the
    rotations that carry the readout axis onto them.
    """
    from scipy.spatial.transform import Rotation

    angles = np.asarray(angles, dtype=float)
    if angles.ndim == 3:
        return [Rotation.from_matrix(matrix) for matrix in angles]
    if angles.ndim == 2:
        readout = np.array([1.0, 0.0, 0.0])
        return [Rotation.align_vectors(direction, readout)[0] for direction in angles]
    return [Rotation.from_euler(axis, float(angle)) for angle in angles]


def _arm(module, index: int):
    """One arm's blocks, or the whole module for a readout with only one."""
    arm = getattr(module, "arm", None)
    return arm(index) if callable(arm) else module.blocks


def _rotated(module, angles, axis: str, kz=None):
    """Play one arm per angle, turned by a rotation extension.

    The rotation rides the readout blocks only: an arm's excitation is played
    in the logical frame whatever the arm is turned to, which is exactly how
    a non-Cartesian loop drives these modules. A stack scales its partition
    encode per arm on top, because that axis is Cartesian and is not turned.
    """
    from pulserver import pypulseq as pp

    events = module.events
    encodes = {
        getattr(events, name, None)
        for name in ("gz_pre", "gz_rew")
        if getattr(events, name, None) is not None
    }
    steps = None if kz is None else np.asarray(kz, dtype=float)

    sequence = pp.Sequence(_system(module))
    cache: dict = {}
    for arm, turn in enumerate(_rotations(angles, axis)):
        rotation = pp.make_rotation(turn)
        for block in _arm(module, arm):
            has_rf = any(getattr(e, "type", None) == "rf" for e in block)
            played = [
                _scaled(event, steps[arm % len(steps)], cache)
                if steps is not None and event in encodes
                else event
                for event in block
            ]
            sequence.add_block(*played, *(() if has_rf else (rotation,)))
    return sequence


def _adc_index(times) -> np.ndarray:
    """Which acquisition each ADC sample belongs to, counting from zero.

    Read off the sample times rather than from a count: consecutive samples
    of one window are a dwell apart and the gap to the next window is the
    rest of the echo spacing, so the boundaries are where the step jumps.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return np.zeros(times.size, dtype=int)
    steps = np.diff(times)
    return np.concatenate([[0], np.cumsum(steps > 1.5 * np.median(steps))])


def trajectory(
    module,
    *,
    ky=None,
    kz=None,
    per: str = "echo",
    angles=None,
    rotation_axis: str = "z",
    plane: str | None = None,
    label: str = "echo",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    path: bool = True,
):
    """Draw where a readout's samples land in k-space.

    Parameters
    ----------
    module : pulserver.SequenceModule
        The readout to play.
    ky, kz : array_like, optional
        Phase- and partition-encode scale factors, in ``[-1, 1]``. One per
        echo under ``per="echo"``, one per repetition under ``per="shot"``.
        Without them the module is played exactly as it publishes itself.
    per : {"echo", "shot"}, optional
        Whether ``ky`` and ``kz`` step within one train or across
        repetitions.
    angles : array_like, optional
        Rotations, in radians, for a non-Cartesian module: one arm per angle,
        turned by a ``ROTATIONS`` extension.
    rotation_axis : str, optional
        Euler axis the rotation turns about. ``"z"`` for a plane or a stack.
    plane : {"xy", "xz", "yz"}, optional
        Which two axes to draw. The two the trajectory actually uses by
        default, and a 3D view when it uses all three.
    label : str, optional
        What the colour means: ``"echo"``, ``"shot"``, ``"arm"``.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size, in inches. Taken from what the trajectory spans by
        default, so an equal-aspect picture is not mostly margin.
    path : bool, optional
        Draw the continuous gradient path behind the samples.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if angles is not None:
        sequence = _rotated(module, angles, rotation_axis, kz)
        # Upstream PyPulseq cannot read the rotation extension, and the dense
        # path is the one output that comes from it.
        path = False
    elif ky is None:
        sequence = module.seq
    else:
        sequence = _replay(module, ky, kz, per)
        if per == "shot" and len(np.atleast_1d(ky)) > 1:
            # The path between two repetitions is a spoiler and a rewinder,
            # not encoding: drawing it turns a stack of lines into a lattice.
            path = False

    result = sequence.calculate_kspace(compat=False, dense=path)
    samples = np.asarray(result.k_traj_adc, dtype=float)
    if samples.size == 0:
        raise ValueError(f"{type(module).__name__} acquires nothing to draw")

    index = _adc_index(result.t_adc)

    names = "xyz"
    widest = max(float(np.ptp(samples[a])) for a in range(3))
    used = [a for a in range(3) if np.ptp(samples[a]) > 0.01 * widest]
    if plane is None and len(used) > 2:
        return _trajectory3d(samples, index, label, title, figsize)
    if plane is None:
        used = (used + [a for a in (0, 1, 2) if a not in used])[:2]
    else:
        used = [names.index(c) for c in plane]

    if figsize is None:
        # Equal aspect and a figure of the wrong shape is all margin. Take
        # the height from what the trajectory actually spans.
        spans = [max(float(np.ptp(samples[a])), 1e-9) for a in used]
        width = 5.4 + (0.8 if index.max() else 0.0)
        figsize = (width, float(np.clip(4.6 * spans[1] / spans[0], 2.0, 4.6)) + 0.8)

    figure, axis = plt.subplots(figsize=figsize)
    if path:
        dense = np.asarray(result.k_traj, dtype=float)
        axis.plot(dense[used[0]], dense[used[1]], color=FAINT, lw=0.7, zorder=1)
    axis.scatter(
        samples[used[0]],
        samples[used[1]],
        c=index,
        cmap=ORDINAL,
        s=5,
        linewidths=0,
        zorder=2,
    )
    if 0 < index.max() < 6:
        # Few enough to name: a colour ramp says which came first, a number
        # says which one this is.
        for group in range(index.max() + 1):
            last = np.flatnonzero(index == group)[-1]
            axis.annotate(
                str(group),
                (samples[used[0]][last], samples[used[1]][last]),
                textcoords="offset points",
                xytext=(5, 0),
                va="center",
                fontsize=8,
                color=MUTED,
            )
    axis.set_xlabel(f"$k_{names[used[0]]}$ [1/m]")
    axis.set_ylabel(f"$k_{names[used[1]]}$ [1/m]")
    axis.set_aspect("equal", adjustable="datalim")
    _style(axis)
    if index.max() > 0:
        _colorbar(figure, axis, index, label)
    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    return figure


def _trajectory3d(samples, index, label, title, figsize):
    """The same picture when the trajectory leaves a plane."""
    figure = plt.figure(figsize=figsize or (5.6, 4.8))
    axis = figure.add_subplot(projection="3d")
    axis.scatter(
        samples[0], samples[1], samples[2], c=index, cmap=ORDINAL, s=3, linewidths=0
    )
    axis.set_xlabel("$k_x$ [1/m]", color=MUTED, fontsize=8)
    axis.set_ylabel("$k_y$ [1/m]", color=MUTED, fontsize=8)
    axis.set_zlabel("$k_z$ [1/m]", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane.pane.set_visible(False)
        pane.line.set_color(FAINT)
        pane._axinfo["grid"].update(color=FAINT, linewidth=0.4)
    if index.max() > 0:
        _colorbar(figure, axis, index, label, pad=0.12)
    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    return figure


def excitation_kspace(
    module,
    *,
    plane: str = "xy",
    title: str | None = None,
    figsize: tuple[float, float] = (8.4, 3.4),
):
    """Draw the excitation k-space a multidimensional pulse traverses.

    A pulse played under moving gradients tips a pattern rather than a slab,
    and the pattern is the transform of the envelope deposited along this
    path. Colour runs with time, so the traversal reads in the order it
    happens.

    Parameters
    ----------
    module : pulserver.design.RfModule
        The module whose pulse to draw.
    plane : {"xy", "xz", "yz"}, optional
        Which two axes the path is drawn in.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size, in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pulse = _first_rf(module)
    result = module.calculate_kspace(compat=False, dense=False)
    # The C core's breakpoint grid, not upstream's dense one: upstream cannot
    # read a pulse whose gradients move under it and answers in a frame of
    # its own.
    path = np.asarray(result.k_traj_breakpoints, dtype=float)
    first, second = ("xyz".index(name) for name in plane)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": (1.2, 1.0)}
    )

    times = 1e3 * np.asarray(pulse.t, dtype=float)
    envelope = 1e6 * np.abs(np.asarray(pulse.signal)) / float(_system(module).gamma)
    left.plot(times, envelope, color=SERIES[0], lw=1.2)
    left.fill_between(times, envelope, color=SERIES[0], alpha=0.12, lw=0)
    left.set_xlabel("time [ms]")
    left.set_ylabel(r"$|B_1|$ [$\mu$T]")
    left.set_xlim(times[0], times[-1])
    _style(left, "envelope")

    steps = np.arange(path.shape[1])
    points = np.stack([path[first], path[second]], axis=1)[:, None, :]
    right.add_collection(
        LineCollection(
            np.concatenate([points[:-1], points[1:]], axis=1),
            array=steps[:-1],
            cmap=ORDINAL,
            linewidths=1.2,
        )
    )
    right.autoscale_view()
    right.set_xlabel(f"$k_{plane[0]}$ [1/m]")
    right.set_ylabel(f"$k_{plane[1]}$ [1/m]")
    right.set_aspect("equal", adjustable="datalim")
    _style(right, "excitation k-space")
    _colorbar(figure, right, [0, len(steps) - 1], "sample")

    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.93 if title else 1.0))
    return figure


def order_figure(
    panels,
    coords=None,
    *,
    trains: int | None = None,
    path: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Draw the views an ordering deals into its trains, coloured by echo.

    An ordering decides which view each echo of a train encodes, and that is
    the whole of the T2 weighting a train carries: where the k-space centre
    lands in the train is the effective echo time, and how fast the ordering
    leaves the centre is how much of the decay the image sees.

    Parameters
    ----------
    panels : sequence of tuple
        ``(label, shots)`` per panel, drawn side by side for comparison.
        ``shots`` is one list per train, indexed by echo: each entry is an
        index into ``coords``, a ``(ky, kz)`` pair, or ``None`` for an echo
        with nothing left to encode.
    coords : array_like, optional
        ``(n_views, 2)`` view coordinates, drawn faintly behind each panel as
        the grid the ordering covers. Required when the shots hold indices.
    trains : int, optional
        Draw only the first few trains. All of them by default.
    path : bool, optional
        Join each train in echo order, which reads only while the trains are
        few.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size, in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = list(panels)
    grid = None if coords is None else np.asarray(coords, dtype=float)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=figsize or (4.2 * len(panels) + 1.4, 4.4),
        squeeze=False,
    )

    highest = 0
    for axis, (label, shots) in zip(axes[0], panels, strict=True):
        if grid is not None:
            axis.scatter(
                grid[:, 0], grid[:, 1], s=5, color=FAINT, linewidths=0, zorder=1
            )
        points, echoes = [], []
        for train in shots[: trains or len(shots)]:
            drawn = []
            for echo, view in enumerate(train):
                if view is None:
                    continue
                where = grid[view] if np.ndim(view) == 0 else np.asarray(view, float)
                points.append(where)
                echoes.append(echo)
                drawn.append(where)
            if path and len(drawn) > 1:
                line = np.asarray(drawn)
                axis.plot(line[:, 0], line[:, 1], color=FAINT, lw=0.7, zorder=2)
        points, echoes = np.asarray(points), np.asarray(echoes)
        highest = max(highest, int(echoes.max()))
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=echoes,
            cmap=ORDINAL,
            s=10 if trains is None else 26,
            linewidths=0,
            zorder=3,
        )
        axis.set_xlabel("phase encode $k_y$ [line]")
        axis.set_ylabel("partition encode $k_z$ [line]")
        axis.set_aspect("equal", adjustable="box")
        _style(axis, label)

    _colorbar(figure, axes[0][-1], [0, highest], "echo")
    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.93 if title else 1.0))
    return figure
