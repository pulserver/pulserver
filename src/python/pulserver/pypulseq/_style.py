"""One look for every figure Pulserver draws.

Matplotlib's defaults are built for exploratory plotting: saturated primaries,
a full box, a legend dropped wherever it fits. A figure that goes into a
report or a documentation page wants the opposite -- the data prominent, the
frame recessive, and the legend somewhere it cannot land on top of a trace.

Nothing here draws anything. These are the colours, the axis treatment and
the legend placement the drawing code applies, kept in one place so that
sequence diagrams, safety plots and magnetisation profiles read as one set.
"""

from __future__ import annotations

__all__ = [
    "FAINT",
    "INK",
    "MAGNITUDE",
    "MUTED",
    "ORDINAL",
    "SAMPLING",
    "SERIES",
    "SIGNED",
    "axis_style",
    "figure_title",
    "image_style",
    "legend_below",
]

from matplotlib.colors import LinearSegmentedColormap

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

#: One hue, light to dark, for an ordered quantity -- an echo index, a shot
#: number. The lightest step still reads against white paper.
ORDINAL = LinearSegmentedColormap.from_list(
    "pulserver-ordinal",
    ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"],
)

#: An acquisition order, first to last. A rainbow rather than one hue: a
#: sampling picture is read for *which* view came when, and a reader picks a
#: position out of a rainbow far more accurately than a step of lightness.
SAMPLING = "turbo"

#: A magnitude that starts at nothing -- the same hue as :data:`ORDINAL`,
#: taken down to paper so that zero reads as blank rather than as pale blue.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "pulserver-magnitude",
    ["#ffffff", "#cfe1f7", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"],
)

#: A signed quantity about zero, for a component that inverts: the ordinal hue
#: one way, its complement the other, paper in between.
SIGNED = LinearSegmentedColormap.from_list(
    "pulserver-signed",
    ["#8a3a12", "#eb6834", "#f7c9b3", "#ffffff", "#cfe1f7", "#2a78d6", "#123f78"],
)


def axis_style(axis, title: str = "", *, grid: bool = False) -> None:
    """Recessive frame: two spines, ticks that do not shout.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        What to restyle.
    title : str, optional
        Set as a left-aligned title.
    grid : bool, optional
        Keep a hairline grid, for a panel read against its axis values rather
        than by shape.
    """
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(FAINT)
    axis.set_facecolor("none")
    axis.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    if grid:
        axis.grid(True, color=FAINT, linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    else:
        axis.grid(False)
    if title:
        axis.set_title(title, loc="left", fontsize=9, color=INK)


def image_style(axis, title: str = "") -> None:
    """Like :func:`axis_style`, but a heatmap keeps all four sides of its frame."""
    for spine in axis.spines.values():
        spine.set_color(FAINT)
    axis.grid(False)
    axis.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    if title:
        axis.set_title(title, loc="left", fontsize=9, color=INK)


def figure_title(figure, text: str | None) -> None:
    """A left-aligned figure title, or nothing when there is no text."""
    if text:
        figure.suptitle(text, x=0.01, ha="left", fontsize=10, color=INK)


def legend_below(figure, handles, labels, *, columns: int | None = None) -> None:
    """Put one legend under the whole figure, in a single row where it fits.

    A legend inside the axes lands on the data sooner or later -- and on a
    safety plot the trace it hides is the one being judged. Below the figure
    it never can, and the panels keep the aspect they were laid out with.
    """
    if not handles:
        return
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=columns or min(len(handles), 5),
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
    )
