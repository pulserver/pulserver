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

from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

__all__ = [
    "MODELS",
    "ORDINAL",
    "brain",
    "context_example",
    "ixi_stack",
    "learned_example",
    "SAMPLING",
    "SERIES",
    "excitation_kspace",
    "images",
    "epi_example",
    "koosh_spokes",
    "spiral_projections",
    "noncartesian_example",
    "order_figure",
    "phantom",
    "radial_spokes",
    "recon_example",
    "sampling",
    "slab_example",
    "stack_example",
    "stack_of_stars",
    "trajectory",
    "volume",
    "volume_example",
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
    return Phantom(image, _coil_ring(size, coils)[None])


#: Where ``docs/_bench/train_denoiser.py`` writes the bundle a figure
#: reconstructs with. A figure names the model; this is the search path.
MODELS = Path(__file__).resolve().parent.parent / "_models"


#: Where the loop elements sit, in units of the field of view, and how big
#: they are. The ring stands just outside the object it surrounds.
_COIL_DISTANCE = 0.6
_COIL_RADIUS = 0.2
_COIL_SEGMENTS = 50


def _coil_ring(size: int, coils: int):
    """Receive sensitivities of a ring of loop elements around the object.

    Each element is a circular current loop, and what it receives at a point
    is the transverse field it would produce there -- ``b_x + i b_y`` by
    Biot-Savart, summed over the segments the loop is drawn with. The phase
    that carries is the whole point: an array whose maps are real has no
    coil-to-coil phase for a parallel-imaging solve to unfold, and undoes an
    aliased scan visibly worse than a physical one does.
    """
    import math

    import torch

    axis = torch.linspace(-0.5, 0.5, size + 1, dtype=torch.float64)[:-1]
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack(
        [columns.reshape(-1), rows.reshape(-1), torch.zeros(size * size, dtype=torch.float64)],
        dim=-1,
    )
    turn = (
        2.0
        * math.pi
        * torch.arange(_COIL_SEGMENTS, dtype=torch.float64)
        / _COIL_SEGMENTS
    )
    sensitivities = []
    for element in range(coils):
        angle = 2.0 * math.pi * element / coils
        centre = torch.tensor(
            [_COIL_DISTANCE * math.sin(angle), _COIL_DISTANCE * math.cos(angle), 0.0],
            dtype=torch.float64,
        )
        # The loop lies in the plane its normal -- the radial direction -- is
        # perpendicular to, which z and the tangential direction span.
        along = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        across = torch.tensor(
            [math.cos(angle), -math.sin(angle), 0.0], dtype=torch.float64
        )
        curve = centre + _COIL_RADIUS * (
            torch.cos(turn)[:, None] * along + torch.sin(turn)[:, None] * across
        )
        segment = torch.roll(curve, -1, dims=0) - curve
        offset = points[:, None, :] - curve[None, :, :]
        distance = offset.norm(dim=-1).clamp_min(1e-9)
        field = (
            torch.cross(segment.expand_as(offset), offset, dim=-1)
            / distance[..., None] ** 3
        ).sum(1) / (4.0 * math.pi)
        sensitivities.append(
            torch.complex(field[:, 0], field[:, 1]).reshape(size, size)
        )
    stacked = torch.stack(sensitivities).to(torch.complex64)
    return stacked / stacked.abs().pow(2).sum(0).sqrt().clamp_min(1e-12)


def brain(size: int = 160, coils: int = 4):
    """A brain slice and a ring of receive coils around it.

    The object is a complex fastMRI brain slice DeepInverse distributes, cropped
    about its centre, so a picture of a learned reconstruction is made on the
    kind of anatomy the model was trained for rather than on a phantom. The
    sensitivities are the analytic ring :func:`phantom` uses.

    Parameters
    ----------
    size : int, optional
        Matrix size, square, cropped from the 320-by-320 slice.
    coils : int, optional
        Elements in the ring.

    Returns
    -------
    Phantom
        ``image`` is a complex ``(1, size, size)`` slice scaled to a unit
        maximum; ``coil_maps`` carries a leading batch.
    """
    import torch
    from deepinv.utils import load_example

    slab = load_example("demo_mini_subset_fastmri_brain_0.pt")
    start = (slab.shape[-1] - size) // 2
    window = slice(start, start + size)
    cropped = slab[..., window, window]
    image = torch.complex(cropped[:, 0], cropped[:, 1])
    image = image / image.abs().max()
    return Phantom(image, _coil_ring(size, coils)[None])


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

    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

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


def radial_spokes(size: int, spokes: int, *, angle_scheme: str = "uniform"):
    """The spokes :func:`~pulserver.app.gre_radial2D_sequence` draws.

    Designed, then read back the way a reconstruction reads it: the sequence's
    own k-space, normalised to MRI-NUFFT's ``[-0.5, 0.5)``. So the picture a
    reconstruction docstring draws is of the trajectory the design side would
    actually have put on the scanner.

    Parameters
    ----------
    size : int
        Readout matrix, which is the samples per spoke.
    spokes : int
        Spokes over half a turn.
    angle_scheme : str, optional
        ``"uniform"`` or ``"golden"``. Uniform is the even spacing a picture
        of a reconstruction wants; golden is what a scan that has to be
        interruptible plays, and its spacing is only quasi-even.

    Returns
    -------
    numpy.ndarray
        ``(spokes, samples, 2)``, float32.
    """
    from pulserver.app import gre_radial2D_sequence

    sequence = gre_radial2D_sequence(
        n_x=size,
        n_spokes=spokes,
        n_slices=1,
        n_dummy=0,
        angle_scheme=angle_scheme,
    )
    samples = np.asarray(sequence.calculate_kspace()[0])[:2]
    samples = samples / (2.0 * np.linalg.norm(samples, axis=0).max())
    return samples.T.reshape(spokes, -1, 2).astype(np.float32)


def _sampled(trajectory, image, coil_maps, coils: int):
    """Measure an object along a trajectory, by forward NUFFT.

    The trajectory's coordinate dimension picks the operator, so a plane and a
    volume are the same call.
    """
    from pulserver.recon import NonCartesian2D, NonCartesian3D

    dimensions = trajectory.shape[-1]
    operator = NonCartesian2D if dimensions == 2 else NonCartesian3D
    physics = operator(
        trajectory.reshape(-1, dimensions),
        tuple(int(size) for size in image.shape[-dimensions:]),
        coil_maps=coil_maps[0],
        n_coils=coils,
    )
    measured = np.asarray(_detached(physics.A(image)))[0]
    return measured.reshape(coils, *trajectory.shape[:-1]).astype(np.complex64)


def volume(size: int = 32, coils: int = 8, depth: int | None = None):
    """The :func:`phantom` given a third dimension.

    The same Shepp-Logan through the slab, tapered along it so the volume has
    structure to resolve in every direction, and the ring of elements extended
    the same way.

    Parameters
    ----------
    size : int, optional
        In-plane matrix, square.
    coils : int, optional
        Elements in the ring.
    depth : int, optional
        Partitions. Half the in-plane matrix by default.

    Returns
    -------
    Phantom
        ``image`` is ``(1, depth, size, size)`` and ``coil_maps`` is
        ``(1, coils, depth, size, size)``.
    """
    import torch

    depth = size // 2 if depth is None else int(depth)
    plane, maps = phantom(size, coils)
    axis = torch.linspace(-1.0, 1.0, depth)
    taper = (1.0 - 0.7 * axis.square()).to(torch.complex64)
    image = plane[0][None] * taper[:, None, None]
    sensitivities = maps[0][:, None] * torch.ones(depth, 1, 1, dtype=torch.complex64)
    return Phantom(image[None], sensitivities[None])


def _view(data, points, *, view: int, coils: int, partition: int = 0, last=False):
    """One non-Cartesian readout, carrying where it was taken."""
    import ismrmrd

    acquisition = ismrmrd.Acquisition()
    acquisition.resize(points.shape[0], coils, trajectory_dimensions=points.shape[1])
    acquisition.data[:] = data.astype(np.complex64)
    acquisition.traj[:] = points
    acquisition.idx.kspace_encode_step_1 = int(view)
    acquisition.idx.kspace_encode_step_2 = int(partition)
    if last:
        acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    return acquisition


def noncartesian_example(
    plugin,
    *,
    size: int = 48,
    coils: int = 8,
    spokes: int | None = None,
):
    """Measure :func:`phantom` along radial spokes and reconstruct it.

    The stream is what a radial scan sends: one readout per spoke, each
    carrying the points it was taken at, and the last one closing the
    measurement. The plugin is driven through the same hooks the inline
    runtime drives, so what comes back is what an online reconstruction
    would return.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive -- a module's ``PLUGIN``, or an instance
        configured differently.
    size : int, optional
        Matrix size, square, and the samples per spoke.
    coils : int, optional
        Elements in the receive array.
    spokes : int, optional
        Spokes acquired. The radial Nyquist count by default, which is what
        sends a scan down the direct branch.

    Returns
    -------
    Measurement
        ``measured`` holds the spokes themselves, ``(spokes, samples, 2)``,
        for :func:`sampling` to draw.
    """
    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

    spokes = int(np.ceil(np.pi / 2 * size)) if spokes is None else int(spokes)
    truth, coil_maps = phantom(size, coils)
    trajectory = radial_spokes(size, spokes)
    readout = trajectory.shape[1]
    measured = _sampled(trajectory, truth, coil_maps, coils)

    stream = [
        _view(
            measured[:, view],
            trajectory[view],
            view=view,
            coils=coils,
            last=view == spokes - 1,
        )
        for view in range(spokes)
    ]
    header = _offline_header(
        size,
        coils,
        spaces=[
            _encoding(
                readout=readout,
                phase_encodes=size,
                views=spokes,
                recon=_matrix(size, size, 1),
                trajectory="RADIAL",
            )
        ],
    )
    result = plugin(AcquisitionBucket(data=tuple(stream)), ReconContext.offline(header))
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)
    return Measurement(
        truth=np.asarray(_detached(truth))[0],
        measured=trajectory,
        # The plugin returns the image in the column/row order it is read in;
        # the phantom is on the (y, x) grid the physics measured.
        image=image.T,
    )


def epi_example(
    plugin,
    *,
    size: int = 48,
    coils: int = 8,
    partitions: int = 1,
    delay: float = 1.0,
    corrected: bool = True,
):
    """Measure :func:`phantom` with an echo-planar train and reconstruct it.

    The stream is what one shot sends: the blip-nulled navigator triplet
    first, then the phase encodes in a single train with every other line
    reversed. A gradient delay puts a linear phase on the reversed lines,
    which is the ghost the navigator is played to remove.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive.
    size : int, optional
        In-plane matrix, square.
    coils : int, optional
        Elements in the receive array.
    partitions : int, optional
        Partitions along the slab axis. One is a plane.
    delay : float, optional
        Gradient delay, in samples, that shifts the reversed lines.
    corrected : bool, optional
        Send the navigator. Without it there is no fit and the reversed lines
        keep their phase, which is what the ghost looks like.

    Returns
    -------
    Measurement
        ``truth`` and ``image`` are the slab's central partition.
    """
    import ismrmrd

    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

    plane = partitions == 1
    truth, coil_maps = phantom(size, coils) if plane else volume(size, coils, partitions)
    object_ = np.asarray(_detached(truth))[0]
    sensitivities = np.asarray(_detached(coil_maps))[0]
    axes = (-2, -1) if plane else (-3, -2, -1)
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(sensitivities * object_, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    ).astype(np.complex64)
    if plane:
        kspace = kspace[:, None]
        object_ = object_[None]

    # A delay shifts a readout along itself, and the train alternates
    # direction, so it is the reversed lines that come back displaced.
    ramp = np.exp(
        2j * np.pi * delay * np.fft.fftshift(np.fft.fftfreq(size))
    ).astype(np.complex64)
    shifted = np.fft.ifft(np.fft.fft(kspace, axis=-1) * ramp, axis=-1)

    def readout(data, *, line, partition=0, flags=(), last=False):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(size, coils)
        acquisition.data[:] = data.astype(np.complex64)
        acquisition.idx.kspace_encode_step_1 = int(line)
        acquisition.idx.kspace_encode_step_2 = int(partition)
        acquisition.center_sample = size // 2
        for flag in flags:
            acquisition.setFlag(getattr(ismrmrd, flag))
        if last:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        return acquisition

    stream = []
    if corrected:
        # Blip-nulled: three readouts of the same line, alternating direction.
        centre = size // 2
        for index in range(3):
            backwards = index % 2 == 1
            flags = ["ACQ_IS_PHASECORR_DATA"]
            if backwards:
                flags.append("ACQ_IS_REVERSE")
            line = (shifted if backwards else kspace)[:, 0, centre]
            stream.append(
                readout(
                    line[:, ::-1] if backwards else line,
                    line=centre,
                    flags=tuple(flags),
                )
            )
    for line in range(size):
        backwards = line % 2 == 1
        source = shifted if backwards else kspace
        for partition in range(partitions):
            data = source[:, partition, line]
            stream.append(
                readout(
                    data[:, ::-1] if backwards else data,
                    line=line,
                    partition=partition,
                    flags=("ACQ_IS_REVERSE",) if backwards else (),
                    last=line == size - 1 and partition == partitions - 1,
                )
            )

    header = _offline_header(
        size,
        coils,
        spaces=[
            _encoding(
                readout=size,
                phase_encodes=size,
                partitions=partitions,
                recon=_matrix(size, size, partitions),
            )
        ],
    )
    result = plugin(AcquisitionBucket(data=tuple(stream)), ReconContext.offline(header))
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)
    middle = partitions // 2
    return Measurement(
        truth=object_[middle],
        measured=np.sqrt((np.abs(kspace[:, middle]) ** 2).sum(axis=0)),
        image=(image if plane else image[middle]).T,
    )


def slab_example(
    plugin,
    *,
    size: int = 32,
    coils: int = 8,
    partitions: int = 8,
    acceleration: int = 1,
    n_acs: int = 0,
):
    """Measure :func:`volume` on a 3D Cartesian lattice and reconstruct it.

    The 3D counterpart of :func:`recon_example`: the autocalibration block
    first, its last encode flagged, then the rest of the ``(partition, line)``
    lattice, and the last encode closing the measurement.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive.
    size : int, optional
        In-plane matrix, square.
    coils : int, optional
        Elements in the receive array.
    partitions : int, optional
        Partitions along the slab axis.
    acceleration : int, optional
        Uniform phase-encode undersampling factor.
    n_acs : int, optional
        Fully sampled autocalibration lines at the centre of the phase-encode
        axis.

    Returns
    -------
    Measurement
        ``truth`` and ``image`` are the slab's central partition; ``measured``
        is that partition's k-space.
    """
    import ismrmrd

    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

    truth, coil_maps = volume(size, coils, partitions)
    object_ = np.asarray(_detached(truth))[0]
    sensitivities = np.asarray(_detached(coil_maps))[0]
    axes = (-3, -2, -1)
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(sensitivities * object_, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    ).astype(np.complex64)

    calibration = list(range(size // 2 - n_acs // 2, size // 2 + n_acs // 2))
    lines = sorted(set(range(0, size, acceleration)) | set(calibration))
    ordered = calibration + [line for line in lines if line not in calibration]

    stream = []
    for index, line in enumerate(ordered):
        for partition in range(partitions):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(size, coils)
            acquisition.data[:] = kspace[:, partition, line]
            acquisition.idx.kspace_encode_step_1 = int(line)
            acquisition.idx.kspace_encode_step_2 = int(partition)
            acquisition.idx.segment = int(index >= len(calibration))
            acquisition.center_sample = size // 2
            if line in calibration:
                acquisition.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)
            closing = partition == partitions - 1
            if closing and calibration and index == len(calibration) - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            if closing and index == len(ordered) - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            stream.append(acquisition)

    header = _offline_header(
        size,
        coils,
        spaces=[
            _encoding(
                readout=size,
                phase_encodes=size,
                partitions=partitions,
                recon=_matrix(size, size, partitions),
            )
        ],
    )
    result = plugin(AcquisitionBucket(data=tuple(stream)), ReconContext.offline(header))
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)
    middle = partitions // 2
    sampled = np.zeros_like(kspace)
    sampled[:, :, lines] = kspace[:, :, lines]
    return Measurement(
        truth=object_[middle],
        measured=np.sqrt((np.abs(sampled[:, middle]) ** 2).sum(axis=0)),
        image=image[middle].T,
    )


def koosh_spokes(size: int, spokes: int):
    """A koosh ball: full diameters through k-space, spread over the sphere.

    Directions come from the golden-means spiral, which is what a 3D radial
    sequence uses to keep every prefix of the scan approximately uniform.

    Parameters
    ----------
    size : int
        Readout matrix, which is the samples per spoke.
    spokes : int
        Diameters acquired.

    Returns
    -------
    numpy.ndarray
        ``(spokes, samples, 3)``, float32.
    """
    index = np.arange(spokes)
    height = 1.0 - 2.0 * (index + 0.5) / spokes
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radial = np.sqrt(np.maximum(1.0 - height**2, 0.0))
    directions = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), height], axis=-1
    )
    radius = np.linspace(-0.5, 0.5, size, endpoint=False)
    return (directions[:, None, :] * radius[None, :, None]).astype(np.float32)


def spiral_projections(size: int, projections: int, *, turns: int | None = None):
    """Spiral projection imaging: a spiral disk through the centre, rotated.

    Every interleave is a spiral that starts at the centre of k-space and
    winds outward in a plane through it, and the planes are turned so their
    normals walk the golden-means spiral over the sphere. Each interleave
    therefore fills a disk rather than tracing a line, which is what lets a
    volume be covered by far fewer of them than a radial scan needs spokes,
    and what leaves the centre of k-space densely visited by every one of
    them -- the property a sensitivity calibration lives on.

    Parameters
    ----------
    size : int
        Matrix, which sets how tightly the spiral has to wind.
    projections : int
        Plane orientations acquired.
    turns : int, optional
        Revolutions per interleave. Enough to reach the matrix by default.

    Returns
    -------
    numpy.ndarray
        ``(projections, samples, 3)``, float32.
    """
    if turns is None:
        turns = int(np.ceil(0.75 * size))
    samples = int(np.ceil(np.pi * turns * size / 4))

    index = np.arange(projections)
    height = 1.0 - 2.0 * (index + 0.5) / projections
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radial = np.sqrt(np.maximum(1.0 - height**2, 0.0))
    normal = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), height], axis=-1
    )

    # Two orthonormal directions spanning the plane each normal defines. The
    # helper is swapped near the poles so the cross product cannot vanish.
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (projections, 1))
    helper[np.abs(normal[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second = np.cross(normal, first)

    step = np.linspace(0.0, 1.0, samples)
    reach = 0.5 * step
    angle = 2.0 * np.pi * turns * step
    return (
        (reach * np.cos(angle))[None, :, None] * first[:, None, :]
        + (reach * np.sin(angle))[None, :, None] * second[:, None, :]
    ).astype(np.float32)


def volume_example(
    plugin,
    *,
    size: int = 24,
    coils: int = 4,
    projections: int | None = None,
):
    """Measure :func:`volume` along spiral projections and reconstruct it.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive.
    size : int, optional
        Matrix, cubic.
    coils : int, optional
        Elements in the receive array.
    projections : int, optional
        Plane orientations acquired. Four times the matrix by default, which
        fills the sphere at the sizes a figure is drawn at.

    Returns
    -------
    Measurement
        ``truth`` and ``image`` are the volume's central partition;
        ``measured`` holds the interleaves themselves.
    """
    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

    if projections is None:
        projections = 4 * size
    truth, coil_maps = volume(size, coils, size)
    trajectory = spiral_projections(size, int(projections))
    measured = _sampled(trajectory, truth, coil_maps, coils)

    stream = [
        _view(
            measured[:, view],
            trajectory[view],
            view=view,
            coils=coils,
            last=view == trajectory.shape[0] - 1,
        )
        for view in range(trajectory.shape[0])
    ]
    header = _offline_header(
        size,
        coils,
        spaces=[
            _encoding(
                # A projection readout already traverses all three axes, so
                # the encoded space has views and samples and nothing else.
                # Declaring phase encodes or partitions beside them would have
                # the buffer place every sample once per position of an axis
                # the scan never stepped.
                readout=trajectory.shape[1],
                phase_encodes=1,
                partitions=1,
                views=trajectory.shape[0],
                recon=_matrix(size, size, size),
                trajectory="SPIRAL",
            )
        ],
    )
    result = plugin(AcquisitionBucket(data=tuple(stream)), ReconContext.offline(header))
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)
    middle = size // 2
    return Measurement(
        truth=np.asarray(_detached(truth))[0, middle],
        measured=trajectory,
        image=image[middle].T,
    )


def stack_of_stars(size: int, spokes: int, partitions: int):
    """The in-plane spokes of a stack, and the partition each plane sits at.

    A stack is Cartesian along its axis, so the trajectory a reconstruction
    needs is one plane's -- every partition repeats it.

    Returns
    -------
    numpy.ndarray
        ``(spokes, samples, 2)``, float32.
    """
    del partitions
    return radial_spokes(size, spokes)


def stack_example(
    plugin,
    *,
    size: int = 32,
    coils: int = 8,
    spokes: int | None = None,
    partitions: int = 8,
):
    """Measure :func:`volume` on a stack of stars and reconstruct it.

    The stream is what a stack sends: every plane's spokes at every partition,
    the same in-plane trajectory throughout, and the last readout closing the
    measurement.

    Parameters
    ----------
    plugin : pulserver.ReconPlugin
        The plugin to drive.
    size : int, optional
        In-plane matrix, square, and the samples per spoke.
    coils : int, optional
        Elements in the receive array.
    spokes : int, optional
        Spokes per plane. The radial Nyquist count by default.
    partitions : int, optional
        Planes along the stack axis.

    Returns
    -------
    Measurement
        ``truth`` and ``image`` are the volume's central partition;
        ``measured`` is the in-plane trajectory.
    """
    from pulserver.recon import ReconContext
    from pulserver.mrd import AcquisitionBucket

    spokes = int(np.ceil(np.pi / 2 * size)) if spokes is None else int(spokes)
    truth, coil_maps = volume(size, coils, partitions)
    trajectory = stack_of_stars(size, spokes, partitions)
    readout = trajectory.shape[1]

    # A stack is a Cartesian axis, so its measurement is the plane-wise
    # trajectory applied to every partition of the volume's own transform.
    import torch

    axes = (-3,)
    planes = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(truth, dim=axes), dim=axes, norm="ortho"),
        dim=axes,
    )
    measured = np.stack(
        [
            _sampled(trajectory, planes[:, index], coil_maps[:, :, index], coils)
            for index in range(partitions)
        ],
        axis=1,
    )

    stream = []
    for partition in range(partitions):
        for view in range(spokes):
            stream.append(
                _view(
                    measured[:, partition, view],
                    trajectory[view],
                    view=view,
                    coils=coils,
                    partition=partition,
                    last=partition == partitions - 1 and view == spokes - 1,
                )
            )
    header = _offline_header(
        size,
        coils,
        spaces=[
            _encoding(
                readout=readout,
                phase_encodes=size,
                partitions=partitions,
                views=spokes,
                stack=partitions,
                recon=_matrix(size, size, partitions),
                trajectory="RADIAL",
            )
        ],
    )
    result = plugin(AcquisitionBucket(data=tuple(stream)), ReconContext.offline(header))
    image = np.asarray(result[0].data if isinstance(result, list) else result.data)
    middle = partitions // 2
    return Measurement(
        truth=np.asarray(_detached(truth))[0, middle],
        measured=trajectory,
        image=image[middle].T,
    )


def _matrix(x: int, y: int, z: int = 1):
    """One MRD matrix size."""
    from types import SimpleNamespace

    return SimpleNamespace(matrixSize=SimpleNamespace(x=x, y=y, z=z))


def _encoding(
    *,
    readout: int,
    phase_encodes: int,
    partitions: int = 1,
    slices: int = 1,
    views: int = 0,
    stack: int = 0,
    recon=None,
    trajectory: str | None = None,
):
    """One encoding space of an offline header.

    ``views`` and ``stack`` are the ``kspace_encoding_step_1`` and ``_2``
    limits, which is what a non-Cartesian space is sized from -- its views
    bear no relation to the image matrix, so the limit answers rather than
    the encoded matrix. ``recon`` is the matrix the images come back on,
    which is the encoded one unless the scan oversampled or gridded.
    """
    from types import SimpleNamespace

    limits = SimpleNamespace(
        slice=SimpleNamespace(minimum=0, maximum=slices - 1, center=0)
    )
    if views:
        limits.kspace_encoding_step_1 = SimpleNamespace(
            minimum=0, maximum=views - 1, center=views // 2
        )
    if stack:
        limits.kspace_encoding_step_2 = SimpleNamespace(
            minimum=0, maximum=stack - 1, center=stack // 2
        )
    encoded = _matrix(readout, phase_encodes, partitions)
    space = SimpleNamespace(
        encodedSpace=encoded,
        reconSpace=recon if recon is not None else encoded,
        encodingLimits=limits,
    )
    if trajectory is not None:
        space.trajectory = SimpleNamespace(name=trajectory)
    return space


def _offline_header(size: int, coils: int, *, slices: int = 1, spaces=None):
    """The encoded and reconstructed spaces a plugin sizes its buffers from."""
    from types import SimpleNamespace

    if spaces is None:
        spaces = [
            _encoding(readout=size, phase_encodes=size, slices=slices),
        ]
    return SimpleNamespace(
        encoding=list(spaces),
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


def sampling(
    panels,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Draw a row of trajectories: where each scan put its samples.

    The non-Cartesian counterpart of :func:`images`' k-space panel. A gridded
    scan shows what it left on the grid; a trajectory has no grid, so it shows
    the path itself, coloured in acquisition order.

    Parameters
    ----------
    panels : sequence of tuple
        ``(label, points)`` per panel, ``points`` shaped
        ``(views, samples, dimensions)`` or ``(samples, dimensions)``. Three
        coordinates are drawn on a 3D axis; a scan with more views than the
        eye can separate is thinned to a representative subset.
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size, in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = list(panels)
    figure = plt.figure(figsize=figsize or (2.9 * len(panels), 3.2))
    for position, (label, points) in enumerate(panels, start=1):
        values = np.asarray(_detached(points))
        views = values.reshape(-1, values.shape[-2], values.shape[-1])
        spatial = views.shape[-1]
        axis = figure.add_subplot(
            1,
            len(panels),
            position,
            projection="3d" if spatial == 3 else None,
        )
        drawn = views[:: max(1, len(views) // (120 if spatial == 3 else 220))]
        colours = plt.get_cmap(SAMPLING)(np.linspace(0.0, 1.0, len(drawn)))
        for view, colour in zip(drawn, colours, strict=True):
            axis.plot(*view.T, color=colour, linewidth=0.7)
        axis.set_xlim(-0.55, 0.55)
        axis.set_ylim(-0.55, 0.55)
        axis.set_xticks([])
        axis.set_yticks([])
        if spatial == 3:
            axis.set_zlim(-0.55, 0.55)
            axis.set_zticks([])
            axis.view_init(elev=22, azim=-58)
            for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
                pane.pane.set_visible(False)
                pane.line.set_color(FAINT)
        else:
            axis.set_aspect("equal")
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


def wave_gradients(
    samples: int = 256,
    *,
    cycles: int = 8,
    amplitude: float = 3e-3,
    duration: float = 10e-3,
):
    """Sinusoidal wave-encoding gradients, a quarter period apart.

    Parameters
    ----------
    samples
        ADC samples along the readout.
    cycles
        Periods of the sinusoid across the readout.
    amplitude
        Peak gradient on each encoded axis, in T/m.
    duration
        Readout duration, in seconds.

    Returns
    -------
    gradients : torch.Tensor
        Shape ``(2, samples)`` in T/m: phase axis then partition axis.
    raster : float
        Time between gradient samples, in seconds.
    times : torch.Tensor
        ADC sample times relative to the first gradient sample, in seconds.
    """
    import torch

    raster = duration / (samples - 1)
    times = torch.arange(samples) * raster
    rate = 2 * torch.pi * cycles / duration
    gradients = torch.stack(
        [amplitude * torch.sin(rate * times), amplitude * torch.cos(rate * times)]
    )
    return gradients, raster, times


# ----------------------------------------------------------------------
# Learned reconstruction
# ----------------------------------------------------------------------


def ixi_stack():
    """Return the contiguous brain slices the context adapter denoises.

    The slices are derived once by ``docs/_bench/make_ixi_stack.py`` from
    TorchIO's IXITiny through :class:`~pulserver.recon.IXITiny`, and committed
    beside the model bundles, so drawing the figure reads a file rather than
    fetching from a third-party host.

    Returns
    -------
    torch.Tensor
        ``(slices, rows, columns)``, real, scaled to a unit peak.
    """
    import torch

    return torch.load(MODELS / "ixi-stack" / "slices.pt").float()


def _accelerated(size: int, coils: int, acceleration: int, calibration: int):
    """One slice, and the undersampled Cartesian physics that measures it."""
    import torch

    from pulserver.recon import Cartesian2D

    truth, coil_maps = brain(size, coils=coils)
    image = torch.stack([truth.real, truth.imag], 1)

    mask = torch.zeros(1, 1, size, size)
    mask[..., ::acceleration, :] = 1.0
    first = (size - calibration) // 2
    mask[..., first : first + calibration, :] = 1.0

    physics = Cartesian2D(mask, coil_maps, viewed_as_real=True)
    return truth, image, physics


def _unrolled_from_bundle(iterations: int | None = None):
    """Rebuild the unroll that ``train_unroll.py`` deployed.

    The bundle carries the prior network; the algorithm parameters it was
    trained beside are in the manifest, so the optimizer reassembles from the
    two without a second source of truth.
    """
    import deepinv

    from pulserver.recon import ModelStore, NormalEquationL2, ScaledAdjoint

    bundle = ModelStore([MODELS]).resolve("fastmri-unroll")
    network = bundle.load().eval()
    recorded = bundle.metadata
    learned = recorded["params_algo"]
    return deepinv.optim.PGD(
        data_fidelity=NormalEquationL2(),
        prior=deepinv.optim.PnP(network),
        params_algo={
            "stepsize": learned["stepsize"],
            "g_param": learned["g_param"],
            "lambda": learned["lambda"],
        },
        max_iter=iterations or recorded["max_iter"],
        custom_init=ScaledAdjoint(),
    ).eval()


def learned_example(
    *,
    size: int = 160,
    coils: int = 4,
    acceleration: int = 4,
    calibration: int = 16,
    iterations: int = 16,
):
    """Reconstruct one accelerated Cartesian scan four ways.

    The scan is a fastMRI brain slice measured through the analytic receive
    array, undersampled uniformly with a fully sampled centre. What differs
    between the panels is only what fills in what the scan did not measure:
    nothing, a total-variation prior, a foundation model applied directly, and
    an unroll trained against this physics.

    Parameters
    ----------
    size : int, optional
        Matrix size, square.
    coils : int, optional
        Elements in the receive array.
    acceleration : int, optional
        Uniform phase-encode undersampling factor.
    calibration : int, optional
        Fully sampled lines at the centre.
    iterations : int, optional
        Iterations for the total-variation solve.

    Returns
    -------
    list of tuple
        ``(label, image)`` panels for :func:`images`, the label carrying the
        peak signal-to-noise ratio where there is a truth to measure against.
    """
    import deepinv
    import torch

    from pulserver.recon import TV, ScaledAdjoint, pics

    truth, image, physics = _accelerated(size, coils, acceleration, calibration)
    measured = physics.A(image)

    def decibels(value):
        error = torch.nn.functional.mse_loss(value, image)
        return float(10 * torch.log10(image.abs().max() ** 2 / error))

    def magnitude(value):
        return torch.complex(value[:, 0], value[:, 1])[0].abs()

    with torch.no_grad():
        adjoint = ScaledAdjoint()(measured, physics)
        classical = pics(
            measured, physics, TV(), regularization=0.01, iterations=iterations
        )
        foundation = deepinv.models.RAM(pretrained=True).eval()(measured, physics)
        unrolled = _unrolled_from_bundle()(measured, physics)

    reconstructions = [
        ("zero filled", adjoint),
        ("total variation", classical),
        ("RAM, applied directly", foundation),
        ("unrolled, trained here", unrolled),
    ]
    return [("object", truth[0].abs())] + [
        (f"{label}, {decibels(value):.1f} dB", magnitude(value))
        for label, value in reconstructions
    ]


def context_example(*, sigma: float = 0.12, shown: int = 4):
    """Denoise a stack of brain slices with a two-dimensional network.

    :class:`~pulserver.recon.ContextAgnosticDenoiser` folds whatever axes a
    volume carries above the spatial ones into the batch a 2D network expects,
    so one slice-wise denoiser reaches a whole stack in one call. The slices
    are adjacent, which is what makes the result worth looking at as a volume
    rather than as independent pictures.

    Parameters
    ----------
    sigma : float, optional
        Noise level added to the stack, and the level the denoiser is called
        at.
    shown : int, optional
        Slices drawn, taken from the middle of the stack.

    Returns
    -------
    tuple
        The noisy and denoised stacks, and the peak signal-to-noise ratio of
        each, so a caller draws what it likes from them.
    """
    import torch

    from pulserver.recon import (
        ContextAgnosticDenoiser,
        NoiseConditioned,
        load_model,
    )

    torch.manual_seed(0)
    truth = ixi_stack()
    volume = torch.stack([truth, torch.zeros_like(truth)], 1)
    noisy = volume + sigma * torch.randn_like(volume)

    denoiser = ContextAgnosticDenoiser(
        NoiseConditioned(load_model("fastmri-denoiser", paths=[MODELS])).eval()
    )
    with torch.no_grad():
        cleaned = denoiser(noisy, sigma)

    def decibels(value):
        error = torch.nn.functional.mse_loss(value, volume)
        return float(10 * torch.log10(volume.abs().max() ** 2 / error))

    first = max((truth.shape[0] - shown) // 2, 0)
    window = slice(first, first + shown)
    return (
        noisy[window, 0],
        cleaned[window, 0],
        decibels(noisy),
        decibels(cleaned),
    )
