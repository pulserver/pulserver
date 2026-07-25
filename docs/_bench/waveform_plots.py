#!/usr/bin/env python3
"""Representative-TR waveform and PNS plots for the docs' safety explanations.

Documentation-only tooling, in the spirit of ``bench_zoo.py``/``bench_safety.py``:
produces the per-sequence figures embedded in ``pns_safety.md``,
``mechanical_resonance_safety.md`` and ``sequence_representation.md``. Not part
of the shipped package.

Drives the same five-sequence safety corpus already used for the mechanical-
resonance verdict matrix (``tests/utils/expected/*_1sl_1avg.seq``), through the
public ``pulserver.io.read`` / ``pulserver.pypulseq.Sequence`` API so the
plots are exactly what a sequence author sees from ``Sequence.plot()`` /
``Sequence.pns()`` while authoring.

Each fixture opens with a handful of dummy (non-ADC) TRs to reach steady
state, so ``tr_index=0`` alone is not representative -- ``first_adc_tr()``
walks forward to the first TR that actually plays out an ADC.

Usage:
    /home/mcencini/pulserver-project/.venv/bin/python docs/_bench/waveform_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pulserver.io as pio  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from pypulseq.utils.safe_pns_prediction import safe_example_hw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "utils" / "expected"
OUT_DIR = Path(__file__).resolve().parent.parent / "explanations" / "assets"

# Representative Irnich chronaxie/rheobase, same values already quoted in
# pns_safety.md's own code sample -- a generic body-gradient point, not any
# particular scanner's configuration (see that page's closing note).
PNS_OPTS = dict(chronaxie_us=360.0, rheobase=4.25e8, alpha=0.333)

# Same corpus mechres_plots/aeq_current.py uses (gre/epi/fse/mprage PASS,
# bssfp FAIL), so a reader can line the two sets of figures up TR for TR.
CORPUS = ["gre_2d", "epi_2d", "fse_2d", "mprage_2d", "bssfp_2d"]

# The SAFE panel is drawn only for this subset -- it needs upstream
# pypulseq's own SAFE model, which is a separate code path from the
# chronaxie plots above, and one pairing (GRE, the isolated-event baseline)
# is enough to show what the model does.
SAFE_CORPUS = ["gre_2d"]

# Upstream pypulseq's own bundled hardware description, explicitly documented
# there as "EXAMPLE scanner hardware (not a real scanner)". Not fabricated
# here, and not any particular system's SAFE coefficients.
SAFE_HW = safe_example_hw()


def first_adc_tr(seq: pp.Sequence, max_tr: int = 64) -> int:
    """First TR index whose block range plays out at least one ADC."""
    for i in range(max_tr):
        try:
            block_range = seq.tr_block_range(i)
        except IndexError:
            break
        if block_range is None:
            break
        lo, hi = block_range
        for b in range(lo, hi + 1):
            if seq.get_block(b).adc is not None:
                return i
    return 0


def load(name: str) -> pp.Sequence:
    return pio.read(FIXTURES / f"{name}_1sl_1avg.seq", system=pp.Opts(**PNS_OPTS))


def plot_tr_waveform(name: str, seq: pp.Sequence, tr_index: int) -> Path:
    seq.plot(tr_index=tr_index, stacked=True)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 11)
    fig.suptitle(f"{name}  --  representative TR (index {tr_index}, first with an ADC)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "representative_tr" / f"{name}_tr.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_pns(name: str, seq: pp.Sequence, tr_index: int) -> Path:
    pns_percent, _ = seq.pns(tr_index=tr_index, plot=True)
    fig = plt.gcf()
    fig.suptitle(f"{name}  --  PNS, TR index {tr_index}  (peak {pns_percent.max():.0f} %)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "pns_safety" / f"{name}_pns.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_pns_safe(name: str, seq: pp.Sequence, tr_index: int) -> Path:
    seq.pns(tr_index=tr_index, model="safe", hardware=SAFE_HW)
    fig = plt.gcf()
    fig.suptitle(
        f"{name}  --  SAFE PNS, TR index {tr_index}  "
        "(pypulseq example hardware -- not a real scanner)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "pns_safety" / f"{name}_pns_safe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    for name in CORPUS:
        seq = load(name)
        tr_index = first_adc_tr(seq)
        tr_out = plot_tr_waveform(name, seq, tr_index)
        pns_out = plot_pns(name, seq, tr_index)
        print(f"{name:12} tr_index={tr_index:3d}  -> {tr_out.relative_to(OUT_DIR.parent)}  "
              f"{pns_out.relative_to(OUT_DIR.parent)}")
        if name in SAFE_CORPUS:
            # upstream calculate_pns()'s time_range scoping is unreliable past
            # TR 0 on some fixtures (a pypulseq limitation, not ours) -- TR 0
            # is enough to illustrate the model's shape; this is an authoring
            # visualisation, not a verdict, so it need not be the steady-state
            # ADC TR the chronaxie plot above uses.
            safe_out = plot_pns_safe(name, seq, 0)
            print(f"{'':12} {'':13}    {safe_out.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
