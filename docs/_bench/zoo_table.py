"""Emit the sequence-zoo summary table, in MyST markdown, from zoo_report.json.

Companion to ``zoo_report.py``; run it after a sweep and paste the output into
``docs/explanations/validation/sequence_zoo.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "zoo_report.json"
ASSETS = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "zoo"


def bound_figure() -> None:
    """One chart for the whole zoo: instance PNS peaks against the envelope."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(RESULTS.read_text())
    rows = [
        (name, entry["pns"])
        for name, entry in results.items()
        if "pns" in entry
    ]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.2))
    for i, (name, pns) in enumerate(rows):
        ratios = [p / pns["envelope_peak"] for p in pns["instance_peaks"]]
        ax.plot(ratios, [i] * len(ratios), "o", ms=4, color="#7aa6c2", alpha=0.7)
    ax.axvline(1.0, color="#c23b22", lw=1.6)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows], fontsize=8)
    ax.set_xlabel("peak PNS of each real TR instance, relative to the worst-case envelope")
    ax.set_xlim(0.8, 1.05)
    ax.invert_yaxis()
    ax.set_title("The envelope bounds every instance, and tightly", fontsize=10)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "zoo_pns_bound.png", dpi=120)
    plt.close(fig)


def fmt_blocks(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main() -> None:
    results = json.loads(RESULTS.read_text())
    print(
        "| Family | Blocks | Design | Rate | Serialise | C parse + convert |"
        " TR blocks | Envelope bounds instances | In-band lines |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for name, entry in results.items():
        if "error" in entry:
            print(f"| `{name}` | — | — | — | — | — | — | build failed | — |")
            continue
        large = entry["sizes"][-1]
        conv = entry["conversion"]
        pns = entry["pns"]
        mech = entry["mechres"]
        bound = "yes" if pns["bound_holds"] else "**no**"
        lines = f"{mech['num_lines']} flagged" if not mech["ok"] else "none"
        print(
            f"| `{name}` | {fmt_blocks(large['blocks'])} |"
            f" {large['design_s']:.2f} s | {large['us_per_block']:.1f} µs/block |"
            f" {conv['write_s']:.2f} s | {conv['convert_s'] * 1000:.0f} ms |"
            f" {entry['tr_size']} | {bound} | {lines} |"
        )


if __name__ == "__main__":
    main()
    bound_figure()
