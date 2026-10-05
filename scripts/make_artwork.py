"""Write the pulserver logo, its mark and the static figures into ``docs/_static``.

    python scripts/make_artwork.py

Every image is written twice, ``<name>.svg`` for the light theme and
``<name>-dark.svg`` for the dark one, so the documentation shows the variant
of the theme the reader selected (the ``only-light`` and ``only-dark``
classes) rather than one chosen by the reader's operating system. The text is
SVG text in the reader's sans-serif font; no font is embedded.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static"

#: The colours of each theme, by role.
PALETTE = {
    "light": {
        "box": "#eef4fa",
        "box_edge": "#2b76ad",
        "core": "#2b76ad",
        "core_edge": "#174b70",
        "accent": "#fdf6e6",
        "accent_edge": "#ffb91f",
        "text": "#173042",
        "muted": "#657786",
        "ink": "#12212b",
        "blue": "#3a77a8",
        "amber": "#ffbe2e",
        "word": "#2b76ad",
    },
    "dark": {
        "box": "#13293a",
        "box_edge": "#5b9bd0",
        "core": "#2b76ad",
        "core_edge": "#8fc3e8",
        "accent": "#2a2415",
        "accent_edge": "#ffbe2e",
        "text": "#dce7f0",
        "muted": "#93a7b5",
        "ink": "#e8eef3",
        "blue": "#5b9bd0",
        "amber": "#ffbe2e",
        "word": "#8fc3e8",
    },
}

#: The classes every figure draws with.
STYLE = """
.box{{fill:{box};stroke:{box_edge};stroke-width:2}}
.core{{fill:{core};stroke:{core_edge};stroke-width:2}}
.a{{fill:{accent};stroke:{accent_edge};stroke-width:2}}
.txt{{font:22px Arial,sans-serif;fill:{text}}}
.white{{font:23px Arial,sans-serif;fill:#ffffff;font-weight:bold}}
.small{{font:18px Arial,sans-serif;fill:{text}}}
.label{{font:16px Arial,sans-serif;fill:{muted}}}
.arrow{{stroke:{muted};stroke-width:3;marker-end:url(#m)}}
.head{{fill:{muted}}}
.ink{{fill:none;stroke:{ink};stroke-width:5;stroke-linejoin:miter;stroke-linecap:butt}}
.blue{{fill:{blue}}}
.amber{{fill:{amber}}}
.word{{fill:{word}}}
"""

ARROW_HEAD = (
    '<defs><marker id="m" markerWidth="10" markerHeight="10" refX="9" refY="3" '
    'orient="auto"><path class="head" d="M0,0 L0,6 L9,3 z"/></marker></defs>'
)

#: One bipolar trapezoid pair: positive lobe and negative lobe, then the line.
_GRADIENT = """<path class="{first}" d="M24 0 38-28 81-28 93.1 0Z"/>
<path class="{second}" d="M93.1 0 104 25 118 25 130 0Z"/>
<path class="ink" d="M0 0H24L38-28H81L104 25H118L130 0H158"/>"""

_GRADIENTS = (
    '<g transform="translate(26 0)">\n'
    + _GRADIENT.format(first="blue", second="amber")
    + '\n</g>\n<g transform="translate(0 39)">\n'
    + _GRADIENT.format(first="amber", second="blue")
    + "\n</g>"
)


def _svg(view_box: str, label: str, description: str, body: str, theme: str) -> str:
    """Return an SVG document with the theme's style and a title and description."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" role="img" '
        f'aria-labelledby="t d">\n<title id="t">{label}</title><desc id="d">'
        f"{description}</desc>\n<style>{STYLE.format(**PALETTE[theme])}</style>\n"
        f"{body}\n</svg>\n"
    )


def logo(theme: str) -> str:
    """Draw the two bipolar gradients and the wordmark."""
    body = (
        f'<g transform="translate(24 70)">\n{_GRADIENTS}\n</g>\n'
        '<text x="238" y="116" font-family="Arial,Helvetica,sans-serif" '
        'font-size="112" font-style="italic"><tspan class="word">pul</tspan>'
        '<tspan class="amber" font-weight="700">server</tspan></text>'
    )
    return _svg(
        "0 0 800 176",
        "pulserver",
        "Two bipolar trapezoidal gradient waveforms in blue and amber, followed "
        "by the wordmark pulserver",
        body,
        theme,
    )


def mark(theme: str) -> str:
    """Draw the two bipolar gradients alone, for the sidebar and the favicon."""
    return _svg(
        "0 0 196 114",
        "pulserver",
        "Two bipolar trapezoidal gradient waveforms in blue and amber",
        f'<g transform="translate(6 38)">\n{_GRADIENTS}\n</g>',
        theme,
    )


def _box(
    kind: str, x: float, y: float, width: float, height: float, *lines: str
) -> str:
    """Draw a rounded box with its lines of text centred in it."""
    text = {"core": "white", "box": "txt", "a": "small"}[kind]
    step = 25
    first = y + height / 2 - step * (len(lines) - 1) / 2 + 7
    return (
        f'<rect class="{kind}" x="{x}" y="{y}" rx="12" width="{width}" height="{height}"/>'
        + "".join(
            f'<text class="{text}" x="{x + width / 2:g}" y="{first + i * step:g}" '
            f'text-anchor="middle">{line}</text>'
            for i, line in enumerate(lines)
        )
    )


def _arrow(
    x1: float, y1: float, x2: float, y2: float, label: str = "", side: str = "right"
) -> str:
    """Draw an arrow, with a label beside its middle."""
    out = f'<line class="arrow" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
    if label:
        x = x1 + (10 if side == "right" else -10)
        anchor = "" if side == "right" else ' text-anchor="end"'
        out += f'<text class="label" x="{x}" y="{(y1 + y2) / 2 + 5:g}"{anchor}>{label}</text>'
    return out


def architecture(theme: str) -> str:
    """Draw the two services between the scanner and the engines, and the design store."""
    body = "\n".join(
        (
            ARROW_HEAD,
            _box(
                "core",
                120,
                20,
                760,
                70,
                "Scanner interpreter and reconstruction client",
            ),
            _arrow(190, 90, 190, 147, "protocol", "left"),
            _arrow(250, 150, 250, 93, "resolved protocol, design id"),
            _arrow(750, 90, 750, 147, "MRD raw data", "left"),
            _arrow(810, 150, 810, 93, "images"),
            _box("box", 40, 150, 420, 65, "Design calls · pulserver.host"),
            _box("box", 540, 150, 420, 65, "Reconstruction proxy · pulserver.proxy"),
            _arrow(140, 215, 140, 252),
            _arrow(355, 215, 355, 252),
            _arrow(645, 215, 645, 252),
            _arrow(860, 215, 860, 252),
            _box("a", 40, 255, 200, 85, "Scanner-sequence", "plugins"),
            _box("a", 260, 255, 200, 85, "pypulseqpp design,", "IR conversion"),
            _box("a", 540, 255, 200, 85, "Enrichment from", "the sequence"),
            _box("a", 760, 255, 200, 85, "ReconPlugin workers,", "bartorch"),
            _arrow(360, 340, 360, 387, "writes or pushes"),
            _arrow(640, 390, 640, 343, "reads", "left"),
            _box(
                "box",
                150,
                390,
                700,
                60,
                "design store · one directory per design: .seq files, IR cache",
            ),
        )
    )
    return _svg(
        "0 0 1000 470",
        "pulserver architecture",
        "The scanner interpreter exchanges protocols with the design calls and "
        "streams raw data to the reconstruction proxy; the design calls write "
        "designs to the design store, or push them to the proxy's own, and the "
        "proxy reads them",
        body,
        theme,
    )


#: Every image written, by the stem of its file.
ARTWORK = {
    "pulserver-logo": logo,
    "pulserver-mark": mark,
    "architecture": architecture,
}


def main() -> None:
    for stem, draw in ARTWORK.items():
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            path = OUT / f"{stem}{suffix}.svg"
            path.write_text(draw(theme))
            print(f"wrote {path.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
