"""Write the pulserver logo, its mark and the static figures into ``docs/_static``.

    python scripts/make_artwork.py

Every image is written twice, ``<name>.svg`` for the light theme and
``<name>-dark.svg`` for the dark one, so the documentation shows the variant
of the theme the reader selected (the ``only-light`` and ``only-dark``
classes) rather than one chosen by the reader's operating system. The text is
SVG text in the reader's sans-serif font; no font is embedded.
"""

from __future__ import annotations

import math
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
.title{{font:bold 20px Arial,sans-serif;fill:{text}}}
.wave{{fill:none;stroke:{blue};stroke-width:3;stroke-linejoin:round}}
.rf{{fill:none;stroke:{amber};stroke-width:3;stroke-linejoin:round}}
.adc{{fill:{amber}}}
.base{{fill:none;stroke:{muted};stroke-width:1.5}}
.bracket{{fill:none;stroke:{muted};stroke-width:2}}
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


def _sinc(x0: float, y0: float, width: float, height: float) -> str:
    """Return the path of a three-lobe sinc of the given width, peaking up at ``height``."""
    points = []
    for i in range(61):
        u = -3 + 6 * i / 60
        value = 1.0 if u == 0 else math.sin(math.pi * u) / (math.pi * u)
        points.append(f"{x0 + width * i / 60:.1f},{y0 - height * value:.1f}")
    return f'<polyline class="rf" points="{" ".join(points)}"/>'


def _trapezoid(
    x0: float, y0: float, width: float, height: float, css: str = "wave"
) -> str:
    ramp = width / 5
    return (
        f'<polyline class="{css}" points="{x0:g},{y0:g} {x0 + ramp:g},{y0 - height:g} '
        f'{x0 + width - ramp:g},{y0 - height:g} {x0 + width:g},{y0:g}"/>'
    )


def _base_block(y: float, name: str, body: str) -> str:
    """Draw one base block: its name and its normalised waveforms over a baseline."""
    return (
        f'<rect class="box" x="40" y="{y}" rx="10" width="230" height="62"/>'
        f'<text class="txt" x="58" y="{y + 39}">{name}</text>'
        f'<line class="base" x1="105" y1="{y + 48}" x2="255" y2="{y + 48}"/>' + body
    )


def pulseg(theme: str) -> str:
    """Draw base blocks, the virtual segments that list them, and the execution stream."""
    stream = "".join(
        _box("core" if seg == "S1" else "a", 640 + 58 * i, 105, 52, 46, seg)
        for i, seg in enumerate(("S1", "S1", "S1", "S2", "S1"))
    )
    body = "\n".join(
        (
            ARROW_HEAD,
            '<text class="title" x="155" y="36" text-anchor="middle">Base blocks</text>',
            '<text class="title" x="475" y="36" text-anchor="middle">Virtual segments</text>',
            '<text class="title" x="805" y="36" text-anchor="middle">Execution stream</text>',
            _base_block(
                60,
                "B1",
                _sinc(130, 98, 70, 26) + _trapezoid(112, 108, 106, 12),
            ),
            _base_block(
                135,
                "B2",
                _trapezoid(110, 183, 140, 26)
                + '<rect class="adc" x="140" y="187" width="80" height="6"/>',
            ),
            _base_block(210, "B3", _trapezoid(140, 258, 80, 30)),
            _base_block(285, "B4", '<text class="label" x="150" y="327">delay</text>'),
            '<text class="label" x="155" y="372" text-anchor="middle">'
            "normalised waveforms, deduplicated</text>",
            _arrow(275, 180, 370, 180),
            _box("box", 375, 80, 200, 90, "S1", "B1 · B2 · B3"),
            _box("box", 375, 215, 200, 70, "S2  ·  B4"),
            '<text class="label" x="475" y="320" text-anchor="middle">'
            "ordered lists of base-block IDs</text>",
            _arrow(580, 150, 632, 130),
            stream,
            '<text class="label" x="930" y="135">…</text>',
            '<line class="bracket" x1="666" y1="155" x2="700" y2="195"/>',
            _box(
                "a",
                640,
                195,
                330,
                130,
                "one segment instance:",
                "amplitude scales s,",
                "RF and ADC phase and frequency,",
                "rotation R, block durations",
            ),
            '<text class="txt" x="500" y="400" text-anchor="middle">'
            "physical gradient  g(t) = R · s · ĝ(t)</text>",
        )
    )
    return _svg(
        "0 0 1000 420",
        "Scanner representation",
        "Deduplicated base blocks with normalised waveforms; virtual segments as "
        "ordered lists of base-block IDs; the execution stream as a row of "
        "segment instances, each holding its amplitude scales, RF and ADC phase "
        "and frequency, rotation and block durations",
        body,
        theme,
    )


def repetition(theme: str) -> str:
    """Draw one repetition of three segments tiled over the execution stream."""
    kinds = {"S1": "core", "S2": "box", "S3": "a"}
    boxes, brackets = [], []
    for r in range(4):
        x0 = 40 + r * 214 + (40 if r == 3 else 0)
        for i, seg in enumerate(("S1", "S2", "S3")):
            boxes.append(_box(kinds[seg], x0 + 68 * i, 40, 62, 46, seg))
        label = f"repetition {r + 1}" if r < 3 else "repetition N"
        brackets.append(
            f'<polyline class="bracket" points="{x0},{96} {x0},{104} '
            f'{x0 + 198},{104} {x0 + 198},{96}"/>'
            f'<text class="label" x="{x0 + 99}" y="{126}" text-anchor="middle">{label}</text>'
        )
    body = "\n".join(
        (
            ARROW_HEAD,
            *boxes,
            '<text class="txt" x="694" y="70" text-anchor="middle">…</text>',
            *brackets,
            _arrow(500, 140, 500, 178),
            _box(
                "box",
                160,
                182,
                680,
                86,
                "segment IDs, run-length encoded over one period: (S1 S2 S3) &#215; N",
                "each instance keeps its own parameters",
            ),
        )
    )
    return _svg(
        "0 0 1000 290",
        "A repetition tiled over the execution stream",
        "Three virtual segments S1, S2 and S3 form one repetition, which is "
        "played N times; the segment IDs of the stream are stored once per "
        "period as (S1 S2 S3) times N, and each instance keeps its own parameters",
        body,
        theme,
    )


#: Every image written, by the stem of its file.
ARTWORK = {
    "pulserver-logo": logo,
    "pulserver-mark": mark,
    "architecture": architecture,
    "pulseg": pulseg,
    "repetition": repetition,
}


def main() -> None:
    for stem, draw in ARTWORK.items():
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            path = OUT / f"{stem}{suffix}.svg"
            path.write_text(draw(theme), encoding="utf-8")
            print(f"wrote {path.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
