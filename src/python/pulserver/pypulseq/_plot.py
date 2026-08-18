"""Laying upstream's two plot windows out as one.

Unstacked, upstream draws RF/ADC on one figure and the gradients on another,
and opens them as two windows. Reading a sequence means reading the two
together, so they are laid out side by side instead -- by moving the axes
upstream made onto a figure of ours, which leaves its plotting code untouched
and its ``SeqPlot`` handles pointing at real axes.
"""

from __future__ import annotations


def _is_merged(plot: object) -> bool:
    """Whether ``plot`` is one of ours, already laid out in two columns."""
    figure = getattr(plot, "fig1", None)
    return (
        figure is not None
        and figure is getattr(plot, "fig2", None)
        and not getattr(plot, "stacked", False)
    )


def _adopt(axis, figure, spec) -> None:
    """Move ``axis`` onto ``figure``, at the grid position ``spec``."""
    axis.remove()
    axis.figure = figure
    figure.add_axes(axis)
    axis.set_subplotspec(spec)


def _merge_columns(plot, *, show_guides: bool = False) -> None:
    """Lay ``plot``'s two figures out as one, three rows by two columns.

    Sharing the time axis has to be re-established by hand: moving an axis
    between figures drops it out of the shared group, silently -- panning one
    column would otherwise leave the other where it was.
    """
    from matplotlib import pyplot as plt

    columns = (tuple(plot.ax1), tuple(plot.ax2))
    sources = (plot.fig1, plot.fig2)
    merged = plt.figure(figsize=(14, 7))
    grid = merged.add_gridspec(3, 2)

    for column, axes in enumerate(columns):
        for row, axis in enumerate(axes):
            _adopt(axis, merged, grid[row, column])

    leader = columns[0][0]
    for axis in (*columns[0][1:], *columns[1]):
        axis.sharex(leader)

    for figure in sources:
        if figure is not None:
            plt.close(figure)

    merged._seq_t_factor = getattr(sources[0], "_seq_t_factor", 1.0)
    merged.tight_layout()

    plot.fig1 = plot.fig2 = merged
    plot.ax1, plot.ax2 = columns
    if show_guides:
        _install_guides(plot)


def _split_columns(plot) -> None:
    """Undo :func:`_merge_columns`, so upstream can draw over ``plot`` again.

    An overlay is upstream's way of comparing two sequences, and it reuses the
    figures of an earlier plot by taking the first three axes of each. Handed
    one merged figure it would take the same three twice and draw both panels
    into the left column, so the columns go back to being two figures for the
    length of the call.
    """
    from matplotlib import pyplot as plt

    merged = plot.fig1
    columns = (tuple(plot.ax1), tuple(plot.ax2))
    figures = []
    for axes in columns:
        figure = plt.figure()
        grid = figure.add_gridspec(3, 1)
        for row, axis in enumerate(axes):
            _adopt(axis, figure, grid[row, 0])
        figure._seq_t_factor = getattr(merged, "_seq_t_factor", 1.0)
        figures.append(figure)

    plt.close(merged)
    plot.fig1, plot.fig2 = figures
    plot.ax1, plot.ax2 = columns


def _install_guides(plot) -> None:
    """Follow the cursor with a hairline across every panel of ``plot``.

    Upstream sets these up too, but binds them to the canvases of the figures it
    made -- which this one replaced. Rebinding is cheaper and less surprising
    than persuading it to draw somewhere else.
    """
    try:
        import mplcursors  # noqa: F401
    except ImportError:
        return

    figure = plot.fig1
    axes = list(dict.fromkeys((*plot.ax1, *plot.ax2)))
    lines = {
        axis: axis.axvline(
            0.0, color="r", linestyle="--", linewidth=1.0, visible=False, zorder=1000
        )
        for axis in axes
    }

    def _follow(event) -> None:
        inside = event.inaxes in axes and event.xdata is not None
        for line in lines.values():
            line.set_visible(inside)
            if inside:
                line.set_xdata([event.xdata])
        figure.canvas.draw_idle()

    plot._vlines = lines
    plot._show_guides = True
    plot._guide_cids = [
        (figure.canvas, figure.canvas.mpl_connect("motion_notify_event", _follow))
    ]
