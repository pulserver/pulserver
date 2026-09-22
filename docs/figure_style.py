"""Matplotlib settings for the gallery's figures.

Every figure is drawn on a transparent canvas in tones that clear 3:1 against
white and against the dark theme's background alike, the palette pypulseqpp's
documentation draws in, so figures of the two projects agree.
``_static/pulserver.css`` removes the card the theme would otherwise paint
behind each image.
"""

from __future__ import annotations

from cycler import cycler

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
}


def gallery_house_style(_gallery_conf, _fname) -> None:
    """Restore the house style after sphinx-gallery has reset matplotlib.

    sphinx-gallery calls ``rcdefaults()`` before each script, so settings made
    in ``conf.py`` do not reach the gallery. Registering this among
    ``reset_modules`` applies them again once the reset has run.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(FIGURE_RCPARAMS)
