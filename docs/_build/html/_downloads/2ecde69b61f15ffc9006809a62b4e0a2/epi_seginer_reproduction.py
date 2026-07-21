#!/usr/bin/env python3
"""Reproduce Seginer et al. 2508.03220 (Fig.1) with the REAL pulserver mechanical-resonance
engine (pulseg_safety.c, called through the pybind11 binding via pulserver.analysis / SequenceCollection
+ pulserver._ext._pulseg_wrapper._calc_mech_resonances) -- no standalone python re-implementation.

Fig.1 of the paper builds a multi-echo multi-slice EPI gradient-train model with THREE nested
periodicities: a single alternating-trapezoid echo train (ETL lobes at echo spacing ESP), N_TE
repeats of that train every DeltaTE (multi-echo), and N_slice repeats of the resulting group every
DeltaTslice (multi-slice). It shows that whether DeltaTE/DeltaTslice sit exactly on the 2*ESP raster
(collapse to one dominant peak) or off it (spread into a comb of comparable peaks) makes an order-
of-magnitude difference to the acoustic spectrum, despite sub-ms timing changes.

Our C engine never "declares" the TE/slice periodicities the way the paper's Eq.2-3 does (as
explicit convolutions with Dirac combs) -- Phase-1 deliberately deleted that kind of sub-period
decomposition (see mechres-aeq-phase1-done memory). Instead every echo/slice instance is
materialized as its own event inside ONE canonical structural window, and the nested sinc-like
trains EMERGE from the coherent complex sum over those events (Stage 3 of the ratified pipeline).
This script demonstrates that the emergent sum reproduces the paper's cascade (train -> +TE ->
+TE+slice) and its on-/off-raster contrast, and additionally exposes the per-event phasor
decomposition (component_* arrays) the C code keeps internally -- a capability beyond what the
paper's magnitude-only product model shows.

Usage:
  /home/mcencini/pulserver-project/.venv/bin/python mechres_plots/epi_seginer_reproduction.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pypulseq as pp  # noqa: E402

from pulserver._ext._pulseg_wrapper import _calc_mech_resonances, _find_tr  # noqa: E402
from pulserver.analysis import Opts, SequenceCollection  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path("/home/mcencini/pulserver-project")
OUT_DIR = REPO / "mechres_plots/assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/tmp/claude-1000/-home-mcencini-pulserver-project/73e760aa-c57a-422b-b761-3e8d067fb7b2/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

DT = 4.0e-6                # GE gradient raster
HZ_PER_MT = 42576.0        # gamma/1000, Hz/m per mT/m

# --- Fig.1 parameters (ramp/flat rounded to the 4 us raster; ESP 0.52 ms vs paper's 0.53 ms) ---
RAMP_S = 0.16e-3            # 40 samples, exact
FLAT_S = 0.20e-3            # 50 samples, exact
ETL = 54
AMP_HZ = 20.0 * HZ_PER_MT   # 20 mT/m peak readout


def sysopts() -> pp.Opts:
    return pp.Opts(max_grad=50.0, grad_unit="mT/m", max_slew=300.0, slew_unit="T/m/s",
                    grad_raster_time=DT, block_duration_raster=DT,
                    rf_raster_time=2e-6, adc_raster_time=2e-6)


def lobe(amp: float, ramp_s: float, flat_s: float, dt: float) -> np.ndarray:
    nr, nf = round(ramp_s / dt), round(flat_s / dt)
    up = np.linspace(0.0, amp, nr, endpoint=False)
    flat = np.full(nf, amp)
    down = np.linspace(amp, 0.0, nr + 1)[1:]  # exactly nr samples, last one == 0.0
    return np.concatenate([up, flat, down])


def echo_train(amp: float, ramp_s: float, flat_s: float, n_lobes: int, dt: float) -> np.ndarray:
    one = lobe(amp, ramp_s, flat_s, dt)
    return np.concatenate([one if i % 2 == 0 else -one for i in range(n_lobes)])


TRAIN = echo_train(AMP_HZ, RAMP_S, FLAT_S, ETL, DT)
T_ETL = len(TRAIN) * DT   # actual rasterized train duration (avoids rounding drift downstream)
ESP = (RAMP_S * 2 + FLAT_S)
print(f"ESP={ESP*1e3:.3f} ms   T_ETL={T_ETL*1e3:.3f} ms   f_fund=1/(2ESP)={1.0/(2*ESP):.1f} Hz")

# On-2ESP-raster spacing: trains/groups abut exactly (zero gap).
DELTA_TE_ON = T_ETL
GROUP_DUR_ON = 2 * DELTA_TE_ON + T_ETL
DELTA_SLICE_ON = GROUP_DUR_ON

# Off-raster ("arbitrary"): same sub-ms offsets the paper uses (+0.45 ms / +0.44 ms).
DELTA_TE_OFF = T_ETL + 0.45e-3
GROUP_DUR_OFF = 2 * DELTA_TE_OFF + T_ETL
DELTA_SLICE_OFF = GROUP_DUR_OFF + 0.44e-3

assert abs(DELTA_TE_ON / (2 * ESP) - round(DELTA_TE_ON / (2 * ESP))) < 1e-9
assert abs(DELTA_SLICE_ON / (2 * ESP) - round(DELTA_SLICE_ON / (2 * ESP))) < 1e-9
assert abs(DELTA_TE_OFF / (2 * ESP) - round(DELTA_TE_OFF / (2 * ESP))) > 1e-3
assert abs(DELTA_SLICE_OFF / (2 * ESP) - round(DELTA_SLICE_OFF / (2 * ESP))) > 1e-3


def arb_block(wf: np.ndarray, sys: pp.Opts):
    return pp.make_arbitrary_grad(channel="x", waveform=wf.astype(float),
                                   first=float(wf[0]), last=float(wf[-1]),
                                   max_slew=1e12, system=sys)


def build_seq(n_te: int, delta_te: float, n_slice: int, delta_slice: float) -> Path:
    """n_te trains every delta_te, that group repeated n_slice times every delta_slice.

    A single 4 us pad block is appended whenever the structure has >1 block: it deliberately
    breaks exact block-level periodicity (see module docstring) so pulserver's own TR-period
    finder falls through to its "genuine single canonical unit" path instead of auto-detecting
    the TE or slice repeat as an ordinary imaging TR (which would trigger the ratified
    infinite-outer-TR assumption on the wrong, INNER periodicity). This way every echo/slice
    instance is materialized as its own event inside ONE structural window, matching how the
    paper's nested sinc-trains are meant to emerge from a coherent sum, not from an assumed-
    infinite repeat.
    """
    sys = sysopts()
    seq = pp.Sequence(system=sys)
    group_dur = (n_te - 1) * delta_te + T_ETL
    for s in range(n_slice):
        for e in range(n_te):
            seq.add_block(arb_block(TRAIN, sys))
            if e < n_te - 1:
                gap = delta_te - T_ETL
                if gap > 1e-9:
                    seq.add_block(pp.make_delay(round(gap / DT) * DT))
        if s < n_slice - 1:
            gap = delta_slice - group_dur
            if gap > 1e-9:
                seq.add_block(pp.make_delay(round(gap / DT) * DT))
    if n_te * n_slice > 1:
        seq.add_block(pp.make_delay(DT))
    path = SCRATCH / f"epi_seginer_te{n_te}_sl{n_slice}_{'on' if abs(delta_te-DELTA_TE_ON)<1e-9 else 'off'}.seq"
    seq.write(str(path))
    return path


def query(path: Path, max_freq_hz: float, forbidden_bands=()):
    relaxed = Opts(max_grad=50.0, max_slew=300.0, B0=3.0,
                   grad_raster_time=DT, block_duration_raster=DT)
    sc = SequenceCollection(str(path), system=relaxed)
    tr_info = _find_tr(sc._cseq, subsequence_idx=0)
    rd = _calc_mech_resonances(
        sc._cseq, subsequence_idx=0, canonical_tr_idx=0,
        target_resolution_hz=2.0, max_freq_hz=max_freq_hz,
        forbidden_bands=list(forbidden_bands),
        peak_log10_threshold=None, peak_norm_scale=None,
        peak_eps=None, peak_prominence=None,
    )
    return rd, tr_info


def to_mt(hz_per_m):
    return np.asarray(hz_per_m) / HZ_PER_MT


# =====================================================================================
# Part 1 -- cascade reproduction of Fig.1: train-only -> +TE -> +TE+slice, off vs on raster
# =====================================================================================
def part1_cascade():
    f_fund = 1.0 / (2 * ESP)
    fmax = 3.2 * f_fund  # covers 1st..3rd harmonic neighbourhood, as in the paper's Fig.1

    p_train = build_seq(1, DELTA_TE_ON, 1, DELTA_SLICE_ON)
    rd_train, ti_train = query(p_train, fmax)

    rows = [
        ("single echo-train\n(1 TE, 1 slice)", rd_train, rd_train),
        (f"+3 TEs, 1 slice\n(ΔTE off / on raster)",
         query(build_seq(3, DELTA_TE_OFF, 1, DELTA_SLICE_ON), fmax)[0],
         query(build_seq(3, DELTA_TE_ON, 1, DELTA_SLICE_ON), fmax)[0]),
        (f"+3 TEs, 6 slices\n(ΔTE,ΔTslice off / on raster)",
         query(build_seq(3, DELTA_TE_OFF, 6, DELTA_SLICE_OFF), fmax)[0],
         query(build_seq(3, DELTA_TE_ON, 6, DELTA_SLICE_ON), fmax)[0]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, sharey="row")
    for r, (label, rd_off, rd_on) in enumerate(rows):
        for c, (rd, coltitle) in enumerate([(rd_off, "off 2·ESP raster (arbitrary)"),
                                             (rd_on, "on 2·ESP raster")]):
            ax = axes[r][c]
            f = np.asarray(rd["analytical_peak_freqs"])
            a = to_mt(rd["analytical_peak_amp_gx"])

            # Matched analytic envelope: the SAME S_ax(f) transform as the A_eq comb,
            # evaluated on a dense uniform grid instead of only at TR harmonics -- see
            # aeq_current.py. Passes exactly through every stem tip, no rescaling.
            n_env = int(rd.get("num_envelope_bins", 0))
            if n_env > 0:
                f_env = np.asarray(rd["envelope_freqs_hz"])[:n_env]
                sf = to_mt(np.asarray(rd["envelope_amp_gx"])[:n_env])
                ax.plot(f_env, sf, color="#1b6ca8", lw=0.7, alpha=0.25, zorder=0)

            # Authoritative A_eq comb, as stems (true to the assumed-infinite-repeat physics:
            # zero content between harmonics, not a connected line).
            ax.vlines(f, 0.0, a, color="#1b6ca8", lw=0.8, alpha=0.9, zorder=2)
            ax.axvline(f_fund, color="gray", ls=":", lw=0.8)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(coltitle, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{label}\nA_eq [mT/m]", fontsize=9)
    axes[-1][0].set_xlabel("Frequency (Hz)")
    axes[-1][1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        "Reproduction of Seginer et al. 2508.03220, Fig.1 -- nested EPI periodicities,\n"
        "computed by the real pulseg_safety.c engine (analytical_peak_amp_gx, per-TR-harmonic A_eq)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "epi_seginer_fig1_reproduction.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> wrote {out}")


# =====================================================================================
# Part 2 -- native per-event phasor decomposition (beyond the paper's magnitude-only model)
# =====================================================================================
def part2_components():
    f_fund = 1.0 / (2 * ESP)
    p = build_seq(3, DELTA_TE_OFF, 6, DELTA_SLICE_OFF)

    # First pass: locate the TRUE peak of the dense analytical comb near the fundamental
    # (the off-raster comb varies rapidly line-to-line -- see row 3 of Fig.1 -- so we must not
    # just guess f_fund itself).
    rd0, _ = query(p, f_fund + 50.0)
    f0 = np.asarray(rd0["analytical_peak_freqs"])
    a0 = np.asarray(rd0["analytical_peak_amp_gx"])
    near = np.abs(f0 - f_fund) < 40.0
    f_peak_est = float(f0[near][np.argmax(a0[near])])

    # Second pass: a tight forbidden band centered exactly on that peak is used purely as an
    # EXPORT TRIGGER -- component_* arrays are only populated at guarded-band candidate
    # frequencies (see sa_check_structural_violations in pulseg_safety.c); no pass/fail check
    # is implied here.
    band = (f_peak_est - 1.5, f_peak_est + 1.5, 1.0e12)
    rd, ti = query(p, f_fund + 50.0, forbidden_bands=[band])

    cf = np.asarray(rd["candidate_freqs"])
    cga = np.asarray(rd["candidate_grad_amps_gx"])
    i_peak = int(np.argmax(cga))
    f_peak = float(cf[i_peak])

    ax_id = np.asarray(rd["component_axes"])
    cfreq = np.asarray(rd["component_freqs_hz"])
    mask = (ax_id == 0) & (np.abs(cfreq - f_peak) < 1e-3)
    amps = to_mt(np.asarray(rd["component_amps"])[mask])
    phases = np.asarray(rd["component_phases_rad"])[mask]
    contrib = np.asarray(rd["component_contrib_ids"])[mask]

    order = np.argsort(contrib)
    amps, phases, contrib = amps[order], phases[order], contrib[order]
    n_te = 3
    slice_idx = contrib // n_te
    te_idx = contrib % n_te

    z = amps * np.exp(1j * phases)
    cum = np.concatenate([[0], np.cumsum(z)])
    total_mt = float(to_mt(cga[i_peak]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cmap = plt.get_cmap("viridis", 6)
    for k in range(len(z)):
        ax1.bar(k, amps[k], color=cmap(slice_idx[k]), edgecolor="k", linewidth=0.3)
    ax1.set_xlabel("event index k (materialized echo/slice instance)")
    ax1.set_ylabel("|event contribution to S_ax(f_fund)| [mT/m]")
    ax1.set_title(f"18 individual event phasors at f={f_peak:.1f} Hz\n"
                   f"(color = slice index 0-5, 3 TEs each)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 5))
    fig.colorbar(sm, ax=ax1, label="slice index", ticks=range(6))

    ax2.plot(cum.real, cum.imag, "-o", ms=3, color="#c1663a", lw=1.0)
    for k in range(len(z)):
        ax2.annotate("", xy=(cum[k + 1].real, cum[k + 1].imag),
                     xytext=(cum[k].real, cum[k].imag),
                     arrowprops=dict(arrowstyle="->", color=cmap(slice_idx[k]), lw=1.2))
    ax2.plot(0, 0, "k+", ms=10)
    circle = plt.Circle((0, 0), total_mt, fill=False, ls="--", color="gray", lw=0.8)
    ax2.add_patch(circle)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re[S_ax(f_fund)]  [mT/m]")
    ax2.set_ylabel("Im[S_ax(f_fund)]  [mT/m]")
    ax2.set_title(f"Coherent phasor walk: 18 events sum to |S|={total_mt:.2f} mT/m\n"
                   f"(A_eq = (2/T_TR)|S|, T_TR={ti['tr_duration_us']/1e3:.1f} ms)")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Native per-event decomposition (component_* export) -- each of the 18 materialized\n"
        "echo/slice instances contributes one phasor; the coherent vector sum IS the emergent\n"
        "multi-echo/multi-slice comb, with no separate 'TE factor' or 'slice factor' declared.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = OUT_DIR / "epi_seginer_component_decomposition.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> wrote {out}")
    print(f"peak candidate: f={f_peak:.2f} Hz, total A_eq={total_mt:.2f} mT/m, "
          f"n_components={mask.sum()}")


if __name__ == "__main__":
    part1_cascade()
    part2_components()
