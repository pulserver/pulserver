#!/usr/bin/env python3
"""Visual assessment of the RATIFIED A_eq mechanical-resonance criterion (Phase-1 C implementation).

Unlike aeq_prototype.py (the Phase-0 python-only ground truth, superseded in its exact math by
the ratified design -- see the header note added to that file), this script drives the REAL
compiled safety code through pulserver.analysis (SequenceCollection / the pybind11 wrapper around
pulseg_safety.c). Every number plotted here is what predownload actually computes:

  A_eq(f) = (2/T_TR) * |S_ax(f)|    at TR harmonics f = k / T_TR      (analytical_peak_* -- display)
  verdict candidates: same A_eq, evaluated only at harmonics inside [band_lo-guard, band_hi+guard]
  violation iff A_eq > eps = max(band_limit, 0.08 * G_max)

T_TR is the canonical-TR period: for degenerate prep/cooldown sequences this is one imaging TR;
for non-degenerate prep/cooldown (e.g. mprage) it is the whole pass (prep + NEX*imaging + cooldown),
and the outer pass repetition is assumed infinite (Dirac comb at exact k/T_TR -- see guard, not a
finite-N sidelobe pattern).

Usage:
  /home/mcencini/pulserver-project/.venv/bin/python mechres_plots/aeq_current.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pulserver._ext._pulseg_wrapper import _calc_mech_resonances, _find_tr  # noqa: E402
from pulserver.analysis import Opts, SequenceCollection  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path("/home/mcencini/pulserver-project")
FIXTURES = REPO / "pulserver-interpreter/external/pulserver/tests/utils/expected"
ESP_DAT = REPO / "pulserver-interpreter/latest_log/epiespHRMbUHP.dat"
OUT_DIR = REPO / "mechres_plots/assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAM_HZ_PER_G = 4257.6

AXES = ("gx", "gy", "gz")
AXIS_LABEL = {"gx": "Gx", "gy": "Gy", "gz": "Gz"}
COLOR = {"gx": "#1b6ca8", "gy": "#c1663a", "gz": "#4c9a5b"}

# Representative corpus: the S1 set the A_eq criterion was ratified against (2026-07-13).
# mprage is the non-degenerate-prep/cooldown example (T_TR = whole pass, NEX folded inside);
# bssfp is the one genuine FAIL (sustained balanced readout on a TR-fundamental harmonic).
CORPUS = [
    ("gre_2d_1sl_1avg", "PASS"),
    ("epi_2d_1sl_1avg", "PASS"),
    ("fse_2d_1sl_1avg", "PASS"),
    ("mprage_2d_1sl_1avg", "PASS"),
    ("bssfp_2d_1sl_1avg", "FAIL"),
]

MAX_FREQ_HZ = 3000.0
RES_HZ = 5.0


def parse_epiesp(path: Path):
    """Return list of (freq_min_hz, freq_max_hz, amp_hz_per_m, axis_label)."""
    toks = [s.strip() for s in path.read_text().splitlines()
            if s.strip() and not s.strip().startswith(("#", "//"))]
    bands, it = [], iter(toks)
    for ax in ("X", "Y", "Z"):
        try:
            count = int(next(it).split()[0])
        except StopIteration:
            break
        for _ in range(count):
            p = next(it).split()
            esp_min, esp_max, amp_gcm = int(p[0]), int(p[1]), float(p[2])
            bands.append((5.0e5 / esp_max, 5.0e5 / esp_min,
                          amp_gcm * GAM_HZ_PER_G * 100.0, ax))
    return bands


def mt(hz_per_m: float, gamma_hz_per_t: float) -> float:
    return hz_per_m * 1e3 / gamma_hz_per_t


def analyze_one(name: str, expected: str, bands, relaxed: Opts):
    path = FIXTURES / f"{name}.seq"
    sc = SequenceCollection(str(path), system=relaxed)
    gamma = float(sc.system.gamma)
    g_max_hz_per_m = float(relaxed.max_grad)  # pypulseq Opts stores max_grad in Hz/m (SI) already

    c_bands = [(b[0], b[1], b[2]) for b in bands]

    verdict, detail = "n/a", ""
    try:
        sc.check(forbidden_bands=c_bands)
        verdict = "PASS"
    except RuntimeError as e:
        verdict, detail = "FAIL", str(e).replace("\n", " ")[:180]

    tr_info = _find_tr(sc._cseq, subsequence_idx=0)
    rd = _calc_mech_resonances(
        sc._cseq, subsequence_idx=0, canonical_tr_idx=0,
        target_resolution_hz=RES_HZ, max_freq_hz=MAX_FREQ_HZ,
        forbidden_bands=c_bands,
        peak_log10_threshold=None, peak_norm_scale=None,
        peak_eps=None, peak_prominence=None,
    )

    fig, panels = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    freqs = np.asarray(rd["analytical_peak_freqs"])
    max_ga_seen = {"gx": 0.0, "gy": 0.0, "gz": 0.0}

    for pi, ax_name in enumerate(AXES):
        ax = panels[pi]
        color = COLOR[ax_name]
        amps_hz = np.asarray(rd[f"analytical_peak_amp_{ax_name}"])
        amps_mt = mt(amps_hz, gamma)
        if amps_mt.size:
            max_ga_seen[ax_name] = float(np.max(amps_mt))

        # Matched envelope: the SAME closed-form S_ax(f) transform as the A_eq comb
        # (envelope_amp_*), evaluated on a dense uniform grid instead of only at TR
        # harmonics k/T_TR. Because the comb is this same function sampled at k/T_TR,
        # the envelope passes exactly through every stem tip -- no rescaling, no
        # windowing mismatch (unlike the old spectrum_full_*-derived reference).
        n_env = int(rd.get("num_envelope_bins", 0))
        if n_env > 0:
            f_env = np.asarray(rd["envelope_freqs_hz"])[:n_env]
            env_mt = mt(np.asarray(rd[f"envelope_amp_{ax_name}"])[:n_env], gamma)
            ax.plot(f_env, env_mt, color=color, lw=0.8, alpha=0.35, zorder=0,
                    label=f"{AXIS_LABEL[ax_name]} matched analytic envelope")

        # Authoritative quantity: A_eq at exact TR harmonics k/T_TR. Drawn as stems, not a
        # connected line -- under the (ratified) infinite-outer-repeat assumption there is
        # genuinely zero content between harmonics, so interpolating between them is dishonest.
        ax.vlines(freqs, 0.0, amps_mt, color=color, lw=0.9, alpha=0.9, zorder=2,
                  label=f"{AXIS_LABEL[ax_name]}  A_eq at TR harmonics (verdict quantity)")

        for blo, bhi, blim, _bax in bands:
            eps_mt = mt(max(blim, 0.08 * g_max_hz_per_m), gamma)
            ax.axvspan(blo, bhi, color="orange", alpha=0.15, lw=0)
            ax.hlines(eps_mt, blo, bhi, color="red", ls="--", lw=1.0, alpha=0.8)

        cf = np.asarray(rd["candidate_freqs"])
        cga = np.asarray(rd[f"candidate_grad_amps_{ax_name}"])
        # mark every guarded-band harmonic line; red X if it individually exceeds its band's eps
        for b_i, (blo, bhi, blim, _bax) in enumerate(bands):
            eps_here = max(blim, 0.08 * g_max_hz_per_m)
            in_band = (cf >= blo) & (cf <= bhi)
            for i in np.where(in_band)[0]:
                a_hz = cga[i]
                if a_hz <= 0.0:
                    continue
                bad = a_hz > eps_here
                ax.plot(cf[i], mt(a_hz, gamma), marker=("x" if bad else "o"),
                        color=("red" if bad else color), ms=(7 if bad else 4),
                        mew=2 if bad else 1, zorder=5)
                if bad:
                    ax.annotate(f"{mt(a_hz, gamma):.1f}", xy=(cf[i], mt(a_hz, gamma)),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=7, color="red", fontweight="bold")

        ax.set_ylabel(f"{AXIS_LABEL[ax_name]}  A_eq  [mT/m]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    panels[-1].set_xlabel("Frequency (Hz)")
    panels[-1].set_xlim(0, MAX_FREQ_HZ)

    ok = "OK" if verdict == expected else "MISMATCH"
    title = (f"{name}  --  current-code verdict: {verdict} (expected {expected}, {ok})\n"
             f"T_TR={tr_info['tr_duration_us']/1e3:.2f} ms   NEX={tr_info['num_averages']}   "
             f"degenerate_prep={bool(tr_info['degenerate_prep'])}   "
             f"degenerate_cooldown={bool(tr_info['degenerate_cooldown'])}   "
             f"num_passes={tr_info['num_passes']}")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"current_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return dict(name=name, expected=expected, verdict=verdict, ok=(verdict == expected),
                detail=detail, out=out, tr_info=tr_info,
                seq_max_mt=max(max_ga_seen.values()))


def main():
    bands = parse_epiesp(ESP_DAT)
    relaxed = Opts(max_grad=50.0, max_slew=350.0, B0=3.0,
                   grad_raster_time=20e-6, block_duration_raster=20e-6)

    print(f"bands parsed: {len(bands)} from {ESP_DAT.name}")
    rows = []
    for name, expected in CORPUS:
        r = analyze_one(name, expected, bands, relaxed)
        rows.append(r)
        print(f"{r['name']:22} expected={r['expected']:4} got={r['verdict']:4} "
              f"{'OK' if r['ok'] else 'XX'}  seq_max~{r['seq_max_mt']:.2f} mT/m  -> {r['out'].name}")
        if r["detail"]:
            print(f"    {r['detail']}")

    lines = ["# Current-code corpus verdict matrix (real C engine via pulserver.analysis)", ""]
    lines.append(f"{'seq':22} {'expected':8} {'got':8} {'ok':8} {'T_TR(ms)':>9} {'NEX':>4} "
                  f"{'nd_prep':>8} {'nd_cool':>8}")
    for r in rows:
        ti = r["tr_info"]
        lines.append(
            f"{r['name']:22} {r['expected']:8} {r['verdict']:8} "
            f"{'OK' if r['ok'] else 'XX':8} {ti['tr_duration_us']/1e3:9.2f} {ti['num_averages']:4} "
            f"{str(not ti['degenerate_prep']):>8} {str(not ti['degenerate_cooldown']):>8}"
        )
    report = OUT_DIR.parent / "aeq_current_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\n-> wrote {report}")
    all_ok = all(r["ok"] for r in rows)
    print("ALL VERDICTS MATCH" if all_ok else "MISMATCHES ABOVE")


if __name__ == "__main__":
    main()
