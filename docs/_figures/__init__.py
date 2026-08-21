"""Figures the API docstrings draw.

Documentation-only. The ``.. plot::`` directives embedded in Pulserver's
docstrings import this package; nothing in the shipped wheel does, which is
why it lives beside ``conf.py`` rather than under ``src/``.

Four kinds of picture, one function each:

``excitation_kspace``
    The path a multidimensional pulse deposits its energy along.
``trajectory``
    Where a readout's ADC samples land in k-space, coloured by the echo or
    the shot that acquired them, so an echo train reads as an ordering rather
    than as one shape.
``order_figure``
    The ``(ky, kz)`` views an ordering deals into its trains, coloured by the
    echo they are encoded at.
``images``
    A row of images, for the reconstruction side: what was measured beside
    what was recovered from it, in DeepInverse's own example shape.
    :func:`phantom` supplies the object and the array those examples measure.

Every one of them returns the :class:`~matplotlib.figure.Figure` it drew, so
a directive that wants to add to it can.

Two functions supply what a picture is drawn of rather than drawing one.
:func:`phantom` is the object and the receive array a reconstruction example
measures; :func:`recon_example` measures it the way a plugin's sequence would
and drives that plugin over the result, so the row :func:`images` draws is a
real reconstruction rather than an illustration of one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

__all__ = [
    "ORDINAL",
    "SAMPLING",
    "SERIES",
    "excitation_kspace",
    "images",
    "order_figure",
    "phantom",
    "recon_example",
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

#: An acquisition order, first to last. The same rainbow
#: :meth:`~pulserver.pypulseq.Sequence.plot_kspace` colours a sampling with,
#: so a module's own picture and the finished scan's read the same way.
SAMPLING = "turbo"


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


def _colorbar(figure, axis, values, label: str, pad: float = 0.03, cmap=SAMPLING):
    """A discrete ordinal bar, ticked at every step while they are few."""
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axis, fraction=0.045, pad=pad
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
# Reading a module
# ----------------------------------------------------------------------


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
        cmap=SAMPLING,
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
        samples[0], samples[1], samples[2], c=index, cmap=SAMPLING, s=3, linewidths=0
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
    _colorbar(figure, right, [0, len(steps) - 1], "sample", cmap=ORDINAL)

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
            cmap=SAMPLING,
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


# ----------------------------------------------------------------------
# Reconstruction: an object to measure, and a row of images
# ----------------------------------------------------------------------


class Phantom(NamedTuple):
    """What a reconstruction example measures.

    Attributes
    ----------
    image : torch.Tensor
        The object, complex, shaped ``(1, size, size)`` -- the layout every
        physics in :mod:`pulserver.recon` answers in.
    coil_maps : torch.Tensor
        Sensitivities, complex, shaped ``(1, coils, size, size)``,
        root-sum-of-squares normalised.
    """

    image: object
    coil_maps: object


def phantom(size: int = 64, coils: int = 4):
    """A Shepp-Logan and a ring of receive coils around it.

    The object is DeepInverse's own phantom, so a reconstruction example here
    starts where one of theirs does; the sensitivities are analytic, because
    the point of the picture is the solver rather than the array.

    Parameters
    ----------
    size : int, optional
        Matrix size, square.
    coils : int, optional
        Elements in the ring.

    Returns
    -------
    Phantom
    """
    import torch
    from deepinv.utils.phantoms import generate_shepp_logan

    image = generate_shepp_logan(size).to(torch.complex64)[None]

    axis = torch.linspace(-1.0, 1.0, size)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    angles = 2.0 * torch.pi * torch.arange(coils) / coils
    sensitivities = torch.stack(
        [
            torch.exp(
                -((columns - 0.9 * torch.cos(angle)) ** 2) / 1.2
                - ((rows - 0.9 * torch.sin(angle)) ** 2) / 1.2
            )
            for angle in angles
        ]
    ).to(torch.complex64)
    sensitivities = sensitivities / sensitivities.abs().pow(2).sum(0).sqrt()
    return Phantom(image, sensitivities[None])


class Measurement(NamedTuple):
    """What a reconstruction plugin was fed, and what it made of it.

    Attributes
    ----------
    truth : numpy.ndarray
        The object that was measured.
    measured : numpy.ndarray
        The sampled k-space, coils combined, on the encoded grid -- so the
        pattern the sampling left is visible in it.
    image : numpy.ndarray
        What the plugin returned.
    """

    truth: object
    measured: object
    image: object


def _fft2c(image):
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def recon_example(
    plugin,
    *,
    size: int = 64,
    coils: int = 8,
    acceleration: int = 1,
    n_acs: int = 0,
    n_samples: int | None = None,
):
    """Measure :func:`phantom` on a 2D Cartesian grid and reconstruct it.

    The stream is the one the scanner would send: the autocalibration block
    first with its last line flagged, then the remaining phase encodes, and
    the last line of the scan closing the slice. The plugin is driven through
    the same three hooks the inline runtime drives, so what comes back is what
    an online reconstruction would return.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive -- a module's ``PLUGIN``, or an instance
        configured differently.
    size : int, optional
        Matrix size, square.
    coils : int, optional
        Elements in the receive array.
    acceleration : int, optional
        Uniform phase-encode undersampling factor.
    n_acs : int, optional
        Fully sampled autocalibration lines at the centre. Needed for a
        reconstruction that has to estimate sensitivities.
    n_samples : int, optional
        Readout samples acquired, for a partial echo. The full readout by
        default.

    Returns
    -------
    Measurement
    """
    import ismrmrd

    from pulserver import AcquisitionBucket, ReconContext

    truth, coil_maps = phantom(size, coils)
    object_ = np.asarray(_detached(truth))[0]
    sensitivities = np.asarray(_detached(coil_maps))[0]
    kspace = _fft2c(sensitivities * object_).astype(np.complex64)

    calibration = list(range(size // 2 - n_acs // 2, size // 2 + n_acs // 2))
    lines = sorted(set(range(0, size, acceleration)) | set(calibration))
    ordered = calibration + [line for line in lines if line not in calibration]
    taken = size if n_samples is None else int(n_samples)

    stream = []
    for index, line in enumerate(ordered):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(taken, coils)
        acquisition.data[:] = kspace[:, line, size - taken :]
        acquisition.idx.kspace_encode_step_1 = int(line)
        acquisition.idx.segment = int(index >= len(calibration))
        acquisition.center_sample = taken - size // 2
        if line in calibration:
            acquisition.setFlag(
                ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
                if line % acceleration == 0
                else ismrmrd.ACQ_IS_PARALLEL_CALIBRATION
            )
        if calibration and index == len(calibration) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
        if index == len(ordered) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        stream.append(acquisition)

    context = ReconContext.offline(_offline_header(size, coils))
    result = plugin(AcquisitionBucket(data=tuple(stream)), context)
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)

    sampled = np.zeros_like(kspace)
    sampled[:, lines, size - taken :] = kspace[:, lines, size - taken :]
    return Measurement(
        truth=object_,
        measured=np.sqrt((np.abs(sampled) ** 2).sum(axis=0)),
        # The plugin returns the image in the column/row order it is read in;
        # the phantom is on the (y, x) grid the physics measured.
        image=image.T,
    )


def _offline_header(size: int, coils: int, *, slices: int = 1):
    """The encoded and reconstructed spaces a plugin sizes its buffers from."""
    from types import SimpleNamespace

    space = SimpleNamespace(matrixSize=SimpleNamespace(x=size, y=size, z=1))
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=space,
                reconSpace=space,
                encodingLimits=SimpleNamespace(
                    slice=SimpleNamespace(minimum=0, maximum=slices - 1, center=0)
                ),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=coils),
    )


def images(
    panels,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    log: bool = False,
):
    """Draw a row of images: what was measured beside what was recovered.

    Parameters
    ----------
    panels : sequence of tuple
        ``(label, array)`` per panel. Anything with a magnitude will do --
        NumPy or Torch, complex or real, with any leading singleton axes.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size, in inches.
    log : bool or sequence of bool, optional
        Draw the magnitude on a logarithmic ramp, which is what makes k-space
        legible beside an image. One value per panel, so a mixed row asks for
        it only where it belongs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = list(panels)
    logarithmic = [log] * len(panels) if isinstance(log, bool) else list(log)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=figsize or (2.9 * len(panels), 3.2),
        squeeze=False,
    )
    for axis, (label, array), ramp in zip(axes[0], panels, logarithmic, strict=True):
        values = np.abs(np.asarray(_detached(array), dtype=complex))
        values = values.reshape(values.shape[-2:]) if values.ndim > 2 else values
        if ramp:
            # Three decades on a linear grey ramp, which is what makes the
            # centre of k-space and its first tails legible in one picture.
            ceiling = max(values.max(), 1e-12)
            values = np.log10(np.maximum(values, ceiling * 1e-3) / ceiling)
        axis.imshow(values, cmap="gray", interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(FAINT)
        axis.set_title(label, loc="left", fontsize=9, color=INK)

    _title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.92 if title else 1.0))
    return figure


def _detached(array):
    """``array`` as something NumPy will take, Torch or not."""
    detach = getattr(array, "detach", None)
    return detach().cpu().numpy() if callable(detach) else array
