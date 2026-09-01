#!/usr/bin/env python3
"""Text that stays inside the box it was put in.

Documentation-only tooling; not part of the shipped package. The schematic
scripts beside this one draw boxes in data coordinates and fill them with
prose, and the two are measured in different units: a box is a fraction of
the axes, a line of text is a number of points. Nothing keeps them in step,
so a reworded sentence silently prints past the border it belongs to.

:func:`fit_text` closes that gap. It is handed the room a box has and the
words that go in it, wraps them to that width, and shrinks the type only if
wrapping alone cannot make them fit the height.
"""

from __future__ import annotations

#: Never shrink below this; past it the figure is unreadable and the box
#: wants rewriting or resizing instead.
MIN_FONTSIZE = 7.5


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _span(ax, text, fontsize, **kw):
    """Display-space (width, height) of ``text`` drawn at ``fontsize``."""
    art = ax.text(0, 0, text, fontsize=fontsize, **kw)
    bb = art.get_window_extent(_renderer(ax.figure))
    art.remove()
    return bb.width, bb.height


def _wrap(ax, words, fontsize, width_px, **kw):
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and _span(ax, trial, fontsize, **kw)[0] > width_px:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def fit_text(ax, x, y, text, *, width, height=None, fontsize, wrap=True, **kw):
    """Draw ``text`` at ``(x, y)``, wrapped into ``width`` data units.

    Parameters
    ----------
    ax
        Axes to draw on; its figure must already have a renderer.
    x, y
        Anchor in data coordinates, interpreted through ``ha`` / ``va``.
    text
        The prose. Existing line breaks are ignored: the wrapping is redone
        against the room actually available.
    width, height
        The box's interior, in data units. ``height`` may be omitted, in
        which case the text is wrapped but never shrunk.
    fontsize
        The size to draw at if the text fits; the ceiling otherwise.
    wrap
        Whether the text may be broken across lines. A heading reads better
        set one size smaller than split in two, so titles pass ``False``.

    Returns
    -------
    matplotlib.text.Text
        The artist drawn, at whatever size it ended up.
    """
    fig = ax.figure
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((width, height if height is not None else 1.0))
    width_px = abs(p1[0] - p0[0])
    height_px = abs(p1[1] - p0[1]) if height is not None else None

    size = fontsize
    while True:
        if wrap:
            body = "\n".join(_wrap(ax, text.split(), size, width_px, **kw))
        else:
            body = " ".join(text.split())
        w, h = _span(ax, body, size, **kw)
        fits = w <= width_px and (height_px is None or h <= height_px)
        if fits or size <= MIN_FONTSIZE:
            break
        size = max(MIN_FONTSIZE, size - 0.25)

    return ax.text(x, y, body, fontsize=size, **kw)


def data_extent(ax, artist):
    """``(width, height)`` of an already-drawn artist, in data units."""
    bb = artist.get_window_extent(_renderer(ax.figure))
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return abs(x1 - x0), abs(y1 - y0)
