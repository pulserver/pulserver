#!/usr/bin/env python3
"""Schematic (not data-driven) diagram of the SEQDESC state-machine contract.

This draws the *architecture* -- the event stream a consumer steps through
and the state it advances -- not a simulated signal. There is no shipped
Python decoder or Bloch state-machine simulator to draw a real trace from
yet (only the C-side SEQDESC emission exists, in
csrc/src/structure/pulseg_seqdesc.c and cxx/recon/trajectory_cache_reader.*);
see explanations/reconstruction/simulator.md for why this figure is
intentionally schematic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "explanations" / "assets" / "reconstruction" / "seqdesc_state_machine.png"

EVENT_COLORS = {"WAIT": "#9e9e9e", "RF": "#c0524a", "ADC": "#3977af"}
# One illustrative TR-shaped token sequence -- the *shape* a GRE-like SeqEvent
# stream takes (WAIT, RF, WAIT, ADC x3, WAIT), not values read from any file.
TOKENS = ["WAIT", "RF", "WAIT", "ADC", "ADC", "ADC", "WAIT"]


def main() -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.5, 6.5), height_ratios=[1, 1.4])

    # --- top: illustrative event stream -------------------------------
    ax_top.set_xlim(0, len(TOKENS))
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_top.set_title("SequenceDescription.events -- one SeqEvent per WAIT / RF / ADC (illustrative shape)", fontsize=10)
    for i, tok in enumerate(TOKENS):
        box = FancyBboxPatch(
            (i + 0.08, 0.25), 0.84, 0.5,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="black",
            facecolor=EVENT_COLORS[tok],
            alpha=0.85,
        )
        ax_top.add_patch(box)
        ax_top.text(i + 0.5, 0.5, tok, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if i < len(TOKENS) - 1:
            ax_top.annotate("", xy=(i + 1.02, 0.5), xytext=(i + 0.94, 0.5),
                             arrowprops=dict(arrowstyle="->", color="black"))
    ax_top.text(0.0, 0.08, "timestamp_us increasing  -->", fontsize=8, color="#555555")

    # --- bottom: state-machine loop ------------------------------------
    ax_bottom.set_xlim(0, 10)
    ax_bottom.set_ylim(0, 6)
    ax_bottom.axis("off")
    ax_bottom.set_title("Per-event state advance (the undecoded half: no shipped consumer yet)", fontsize=10)

    nodes = {
        "state": (1.3, 3, "isochromat /\nsignal state", "#666666"),
        "wait": (5.0, 5.0, "WAIT\nfree precession", EVENT_COLORS["WAIT"]),
        "rf": (5.0, 3.0, "RF\namp, phase, freq,\nshim (RfDef library)", EVENT_COLORS["RF"]),
        "adc": (5.0, 1.0, "ADC\nrecord state\n(adc_role, phase)", EVENT_COLORS["ADC"]),
        "out": (8.7, 3, "simulated\nsignal", "#666666"),
    }
    for key, (x, y, label, color) in nodes.items():
        w, h = (1.8, 1.4) if key in ("state", "out") else (2.4, 1.3)
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
                              linewidth=1.4, edgecolor="black", facecolor=color, alpha=0.85)
        ax_bottom.add_patch(box)
        ax_bottom.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    for target in ("wait", "rf", "adc"):
        x, y = nodes["state"][0] + 0.9, nodes["state"][1]
        tx, ty = nodes[target][0] - 1.2, nodes[target][1]
        ax_bottom.add_patch(FancyArrowPatch((x, y), (tx, ty), arrowstyle="->", mutation_scale=14, color="black"))
        ax_bottom.add_patch(FancyArrowPatch((tx, ty), (x, y - 0.15), arrowstyle="->", mutation_scale=14,
                                             color="black", connectionstyle="arc3,rad=0.15"))
    ax_bottom.add_patch(FancyArrowPatch((nodes["adc"][0] + 1.2, nodes["adc"][1] + 0.3),
                                         (nodes["out"][0] - 0.9, nodes["out"][1] - 1.6),
                                         arrowstyle="->", mutation_scale=14, color="black",
                                         connectionstyle="arc3,rad=-0.2"))
    ax_bottom.text(1.3, 1.1, "RF shape library\n(RfDef: mag/phase/time,\nbandwidth, shim tuples)",
                   fontsize=7.5, ha="center", color="#333333")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
