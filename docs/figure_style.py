"""Matplotlib settings for the gallery's figures.

Every figure is drawn on a transparent canvas in tones that clear 3:1 against
white and against the dark theme's background alike, the palette pypulseqpp's
documentation draws in, at the type sizes and resolution of bartorch's, so
figures of the three projects agree. ``_static/pulserver.css`` removes the
card the theme would otherwise paint behind each image. A script sets no font
size, DPI or colour of its own: it imports the colours it needs from here and
sizes its figures as fractions of :data:`PAGE_WIDTH`.
"""

from __future__ import annotations

from cycler import cycler

#: The width of the documentation column, in inches.
PAGE_WIDTH = 7.8

#: Axis furniture.
INK = "#717c8b"
MUTED = "#7d8996"
FAINT = "#7d899659"

#: Categorical hues, assigned in order and never cycled.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#169869",
    "#b47900",
    "#c5678b",
    "#008300",
    "#7d6fd4",
    "#e34948",
)

FIGURE_RCPARAMS = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "savefig.transparent": True,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": FAINT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "grid.color": FAINT,
    "legend.labelcolor": INK,
    "axes.prop_cycle": cycler(color=list(SERIES)),
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.constrained_layout.use": True,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 13,
    "legend.frameon": False,
    "image.cmap": "gray",
    "image.interpolation": "nearest",
}


def gallery_house_style(_gallery_conf, _fname) -> None:
    """Restore the house style after sphinx-gallery has reset matplotlib.

    sphinx-gallery calls ``rcdefaults()`` before each script, so settings made
    in ``conf.py`` do not reach the gallery. Registering this among
    ``reset_modules`` applies them again once the reset has run.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(FIGURE_RCPARAMS)
