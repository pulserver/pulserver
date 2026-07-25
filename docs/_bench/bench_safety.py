"""Mechanical-resonance and PNS analysis across the example zoo.

Documentation-only tooling: it produces the tables in
``docs/explanations/safety/mechanical_resonance.md``,
``docs/explanations/safety/pns.md`` and ``docs/explanations/benchmarks.md``.
Not part of the shipped package.

Run it as::

    python docs/_bench/bench_safety.py --esp <epiesp.dat> --out docs/_bench/safety.json

Everything here goes through the pybind11 bindings around ``pulseg_safety.c``
— the same compiled code the scanner runs at predownload — so a number printed
here and a verdict on the scanner cannot disagree by re-implementation.

For each zoo sequence it reports the canonical-window structure the analysis
keys off (TR period, prep/cooldown degeneracy), the peak equivalent gradient
amplitude the mechanical-resonance criterion sees, the peak PNS the Irnich
model predicts, and how long each analysis takes.

Without ``--esp`` a small synthetic band table is used, so the tables are
reproducible without a vendor lockout file. No vendor table ships with
Pulserver.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOO = REPO_ROOT / "examples" / "sequences"
sys.path.insert(0, str(REPO_ROOT / "python"))

import numpy as np  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from pulserver import UIParam  # noqa: E402
from pulserver._ext._pulseg_wrapper import (  # noqa: E402
    _calc_mech_resonances,
    _calc_pns,
    _check_safety,
    _find_tr,
    _PulseqCollection,
)
from pulserver.io import read_esp_bands  # noqa: E402
from pulserver.pypulseq._safety import chronaxie_kernel  # noqa: E402

MAX_FREQ_HZ = 3000.0
RESOLUTION_HZ = 5.0

# Irnich nerve-model constants. Representative body-gradient values, not a
# specific vendor's: the point of the table is the shape of the response, and
# the real constants come from the scanner's own configuration.
CHRONAXIE_US = 360.0
RHEOBASE_HZ_PER_M_PER_S = 4.25e8
ALPHA = 0.333

# Stand-in forbidden bands, used when no vendor ESP table is supplied. Three
# narrow keep-out ranges spanning the usual acoustic resonance region.
SYNTHETIC_BANDS = [
    (540.0, 640.0, 0.0),
    (960.0, 1080.0, 0.0),
    (1180.0, 1260.0, 0.0),
]

# Small, fast protocol sizes: the analysis cost depends on the complexity of
# one canonical window, not on how many TRs follow it, so a 1-slice, 1-average
# version of each sequence exercises exactly the same code path as the
# clinical-size one.
CASES = [
    ("gre_2d", "GRE", {UIParam.NX: 64, UIParam.NY: 64, UIParam.NSLICES: 1}),
    ("fse_2d", "FSE, ETL 16", {UIParam.NX: 64, UIParam.NY: 64, UIParam.NSLICES: 1, UIParam.ETL: 16}),
    ("bssfp_2d", "bSSFP", {UIParam.NX: 64, UIParam.NY: 64}),
    ("mprage_3d", "MPRAGE", {UIParam.NX: 64, UIParam.NY: 64, UIParam.NSLICES: 16}),
    (
        "gre_noncart_2d",
        "spiral, 48 shots (short readout)",
        {UIParam.NX: 128, UIParam.NUM_SHOTS: 48, UIParam.NSLICES: 1},
    ),
    (
        "gre_noncart_2d",
        "spiral, 4 shots (long readout)",
        {UIParam.NX: 128, UIParam.NUM_SHOTS: 4, UIParam.NSLICES: 1},
    ),
]


def load_plugin(name: str):
    path = ZOO / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_safety_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PLUGIN


def build_seq(plugin_name: str, overrides: dict, out_dir: Path) -> tuple[Path, pp.Opts]:
    plugin = load_plugin(plugin_name)
    opts = pp.Opts()
    protocol = plugin.get_default_protocol(opts)
    for key, value in overrides.items():
        protocol[str(key)]["value"] = value
    verdict = plugin.validate_protocol(opts, protocol)
    if not verdict.get("valid", False):
        raise RuntimeError(verdict.get("info", "invalid protocol"))
    path = out_dir / f"safety_{plugin_name}_{len(overrides)}_{abs(hash(tuple(sorted(map(str, overrides.values())))))}.seq"
    plugin.make_sequence(opts, protocol, str(path))
    return path, opts


def analyse(seq_path: Path, opts: pp.Opts, bands, repeats: int) -> dict:
    collection = _PulseqCollection(
        [seq_path.read_bytes()],
        float(opts.gamma),
        3.0,
        float(opts.max_grad),
        float(opts.max_slew),
        float(opts.rf_raster_time),
        float(opts.grad_raster_time),
        float(opts.adc_raster_time),
        float(opts.block_duration_raster),
        True,
        1,
    )
    tr = _find_tr(collection, subsequence_idx=0)

    # The gate itself: what predownload runs. Mechanical resonance only, so the
    # timing is comparable across sequences with and without PNS parameters.
    gate_s = 1e30
    verdict = "PASS"
    detail = ""
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        try:
            _check_safety(collection, forbidden_bands=list(bands), skip_pns=True)
        except RuntimeError as exc:
            verdict, detail = "FAIL", str(exc).replace("\n", " ")[:160]
        gate_s = min(gate_s, time.perf_counter() - t0)

    # The plotting API: same spectrum, evaluated on a dense display grid too.
    spectra_s = 1e30
    for _ in range(repeats):
        t0 = time.perf_counter()
        spectra = _calc_mech_resonances(
            collection,
            subsequence_idx=0,
            canonical_tr_idx=0,
            target_resolution_hz=RESOLUTION_HZ,
            max_freq_hz=MAX_FREQ_HZ,
            forbidden_bands=list(bands),
        )
        spectra_s = min(spectra_s, time.perf_counter() - t0)

    peak_mt_per_m = 0.0
    for axis in ("gx", "gy", "gz"):
        amps = spectra.get(f"analytical_peak_amp_{axis}") or []
        if len(amps):
            peak_mt_per_m = max(peak_mt_per_m, max(amps) * 1e3 / float(opts.gamma))

    # PNS is the other shape of analysis entirely: the C core materialises the
    # slew-rate waveform of the canonical TR on the gradient raster and hands
    # it back, because the nerve model itself is a vendor callback. The
    # convolution below is the Irnich model, the same arithmetic
    # pulserver_ge_pns.c runs on the scanner.
    pns_s = 1e30
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        pns = _calc_pns(
            collection,
            subsequence_idx=0,
            canonical_tr_idx=0,
            chronaxie_us=CHRONAXIE_US,
            rheobase=RHEOBASE_HZ_PER_M_PER_S,
            alpha=ALPHA,
        )
        pns_s = min(pns_s, time.perf_counter() - t0)

    dt = float(opts.grad_raster_time)
    kernel = chronaxie_kernel(dt, CHRONAXIE_US, RHEOBASE_HZ_PER_M_PER_S, ALPHA)
    n = int(pns["num_samples"])
    # The core returns dG/dt in T/m/s; the Irnich constants are in Hz/m/s.
    gamma = float(opts.gamma)
    t0 = time.perf_counter()
    per_axis = [
        np.convolve(np.asarray(pns[f"slew_{ax}"], dtype=float) * gamma, kernel)[:n] for ax in ("x", "y", "z")
    ]
    model_s = time.perf_counter() - t0
    combined = np.sqrt(sum(component**2 for component in per_axis))
    pns_peak_percent = 100.0 * float(combined.max()) if n else None

    return {
        "tr_ms": tr["tr_duration_us"] / 1e3,
        "tr_size": tr["tr_size"],
        "num_trs": tr["num_trs"],
        "degenerate_prep": bool(tr["degenerate_prep"]),
        "degenerate_cooldown": bool(tr["degenerate_cooldown"]),
        "num_canonical_trs": tr["num_canonical_trs"],
        "verdict": verdict,
        "detail": detail,
        "peak_aeq_mt_per_m": peak_mt_per_m,
        "num_candidates": len(spectra.get("candidate_freqs") or []),
        "gate_ms": gate_s * 1e3,
        "spectra_ms": spectra_s * 1e3,
        "pns_slew_ms": pns_s * 1e3,
        "pns_model_ms": model_s * 1e3,
        "pns_samples": n,
        "pns_peak_percent": pns_peak_percent,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esp", type=Path, default=None, help="Vendor ESP lockout table")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "safety.json")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    if args.esp is not None:
        bands = [(fmin, fmax, amp * 1e-3 * 42577478.0) for fmin, fmax, amp, _ in read_esp_bands(args.esp)]
        band_source = str(args.esp)
    else:
        bands = SYNTHETIC_BANDS
        band_source = "synthetic"

    work_dir = args.work_dir or (Path(__file__).resolve().parent / "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for plugin_name, label, overrides in CASES:
        print(f"{plugin_name:<16} {label}", flush=True)
        try:
            seq_path, opts = build_seq(plugin_name, overrides, work_dir)
            row = {"plugin": plugin_name, "label": label, **analyse(seq_path, opts, bands, args.repeats)}
        except Exception as exc:  # noqa: BLE001
            row = {"plugin": plugin_name, "label": label, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"    {json.dumps(row)}", flush=True)

    args.out.write_text(json.dumps({"bands": band_source, "rows": rows}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
