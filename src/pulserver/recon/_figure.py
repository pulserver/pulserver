"""A matplotlib figure as the pixels of an RGB reconstruction result."""

from __future__ import annotations

__all__ = ["figure_rgb"]

import io
from typing import Any

import numpy as np


def figure_rgb(figure: Any, dpi: float | None = None) -> np.ndarray:
    """Return a matplotlib figure drawn to a ``(y, x, 3)`` ``uint8`` array.

    Pass it as ``ReconResult(figure_rgb(fig), image_type="rgb")`` to send a
    plot, a fit or a report to the scanner as a secondary capture.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure; drawn as ``savefig`` draws it, whatever its canvas.
    dpi : float, default=None
        Resolution to draw at; the figure's own when not given.
    """
    dpi = figure.dpi if dpi is None else dpi
    buffer = io.BytesIO()
    figure.savefig(buffer, format="rgba", dpi=dpi)
    width = round(figure.get_figwidth() * dpi)
    rgba = np.frombuffer(buffer.getvalue(), dtype=np.uint8).reshape(-1, width, 4)
    return rgba[..., :3].copy()
