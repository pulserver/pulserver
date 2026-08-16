#!/usr/bin/env python3
"""Regenerate the mechanical-resonance figures in docs/explanations/safety/.

Every number drawn here comes from the compiled safety core, through
:meth:`pulserver.pypulseq.Sequence.mech_resonances` -- there is no Python
re-implementation of the criterion. The default (``compress_trains=True``)
runs the analysis in exactly the form ``pulseg_check_safety`` uses at
predownload, so the candidate lines and the pass/fail markers in these
figures are literally the ones the scanner decides on.

What is plotted, per axis:

  analytic envelope   A_eq(f) on a dense uniform grid  (envelope_amp_g*)
  TR harmonics        A_eq at every k / T_TR           (analytical_peak_amp_g*)
  candidates          A_eq at the harmonics inside a guarded band, which
                      IS the verdict                   (candidate_grad_amps_g*)
  eps                 dashed limit per band: the vendor limit, or the
                      0.08 * G_max floor for a zero-tolerance band

T_TR is the canonical-TR period: for degenerate prep/cooldown sequences that
is one imaging TR; for non-degenerate prep/cooldown (mprage) it is the whole
pass, with NEX folded in analytically.

Usage:
    <venv>/bin/python docs/_bench/mechres_plots.py [--esp PATH]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pulserver.io  # noqa: E402
from pulserver.pypulseq._safety import read_esp_bands  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent
FIXTURES = PKG / "tests/utils/expected"
OUT_DIR = PKG / "docs/explanations/assets/mechanical_resonance"

# The reference corpus the A_eq criterion was ratified against: four PASS
# cases spanning isolated-readout, echo-train and non-degenerate-prep
# structures, plus the one genuine FAIL (a balanced readout parked on a
# TR-fundamental harmonic).
CORPUS = [
    ("gre", "gre_2d_1sl_1avg", "PASS"),
    ("epi", "epi_2d_1sl_1avg", "PASS"),
    ("fse", "fse_2d_1sl_1avg", "PASS"),
    ("mprage", "mprage_2d_1sl_1avg", "PASS"),
    ("bssfp", "bssfp_2d_1sl_1avg", "FAIL"),
]

# Fallback when no vendor ESP table is supplied. Public synthetic bands, in
# (freq_min_hz, freq_max_hz, max_amplitude_mT_per_m); 0.0 means
# zero-tolerance, so eps falls back to the 0.08 * G_max floor.
DEFAULT_BANDS = [
    (540.0, 640.0, 0.0),
    (960.0, 1080.0, 0.0),
    (1180.0, 1260.0, 0.0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esp", type=Path, default=None, help="vendor ESP lockout table")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    bands = read_esp_bands(args.esp) if args.esp else DEFAULT_BANDS
    args.out.mkdir(parents=True, exist_ok=True)

    rc = 0
    for tag, fixture, expected in CORPUS:
        path = FIXTURES / f"{fixture}.seq"
        if not path.exists():
            print(f"  {tag:8s} SKIP (missing {path.name})")
            rc = 1
            continue

        seq = pulserver.io.read(str(path))
        record = seq.mech_resonances(bands=bands, plot=True)

        # Distinct lines: candidates are enumerated per band, so a harmonic
        # inside two overlapping bands (an ESP table repeating the same row
        # for X and Y does exactly this) is listed once per band.
        freqs = np.round(np.asarray(record["candidate_freqs"], dtype=float), 6)
        viol = np.asarray(record["candidate_violations"], dtype=int)
        n_cand = np.unique(freqs).size
        n_viol = np.unique(freqs[viol != 0]).size
        verdict = "FAIL" if n_viol else "PASS"
        out = args.out / f"current_{tag}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close("all")

        flag = "" if verdict == expected else f"  <-- EXPECTED {expected}"
        print(
            f"  {tag:8s} {n_cand:5d} candidates, "
            f"{n_viol:3d} violating -> {verdict}{flag}  [{out.name}]"
        )
        if verdict != expected:
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
