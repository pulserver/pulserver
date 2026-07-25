"""Runtime and memory of the write_sequence pipeline, across the example zoo.

Documentation-only tooling: it produces the tables in
``docs/explanations/benchmarks.md``. Not part of the shipped package.

Run it as::

    bash docs/_bench/build_bench.sh
    python docs/_bench/bench_zoo.py --out docs/_bench/results.json

Each case is one plugin from ``examples/sequences/`` at one protocol size. For
each case the script times, separately:

===============  =========================================================
``validate``     ``plugin.validate_protocol`` — the call the scanner UI
                 makes on every parameter change
``design``       ``plugin.make_sequence`` minus the serialisation below:
                 building every event and appending every block
``serialise``    ``pulserver.io.write`` up to the payload bytes
``disk``         writing those bytes to the filesystem
===============  =========================================================

then hands the resulting ``.seq`` to ``bench_pipeline`` for the C stages
(parse, convert, cache write, and the three cache loads).

Timings are the minimum of ``--repeats`` runs; the minimum, not the mean,
because scheduler noise only ever adds. Python memory is ``tracemalloc``'s
peak over the design phase, which counts Python objects only — the C stages
report process RSS instead.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOO = REPO_ROOT / "examples" / "sequences"
BENCH_PIPELINE = Path(__file__).resolve().parent / "bench_pipeline"

sys.path.insert(0, str(REPO_ROOT / "python"))

import pulserver.io as pio  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from pulserver import UIParam  # noqa: E402


@dataclass
class Case:
    """One plugin at one protocol size."""

    plugin: str
    label: str
    overrides: dict = field(default_factory=dict)


def _p(key, value):
    """Protocol override entry, keyed the way the plugins read it."""
    return (str(key), value)


# The zoo covered by the performance tables: one Cartesian gradient echo, one
# spin-echo train, one blipped EPI train, one balanced steady state, one
# inversion-prepared 3D, and non-Cartesian with a short and a long spiral
# readout. Sizes span a small prototype, a realistic clinical protocol, and a
# large one, so the tables show how each stage scales rather than a single
# number.
CASES = [
    Case("gre_2d", "64² × 1 slice", dict([_p(UIParam.NX, 64), _p(UIParam.NY, 64), _p(UIParam.NSLICES, 1)])),
    Case("gre_2d", "128² × 16 slices", dict([_p(UIParam.NX, 128), _p(UIParam.NY, 128), _p(UIParam.NSLICES, 16)])),
    Case(
        "gre_2d",
        "256² × 32 slices",
        dict(
            [
                _p(UIParam.NX, 256),
                _p(UIParam.NY, 256),
                _p(UIParam.NSLICES, 32),
                _p(UIParam.BANDWIDTH, 31250.0),
                _p(UIParam.TR, 700.0),
            ]
        ),
    ),
    Case(
        "gre_3d",
        "128² × 64 partitions",
        dict([_p(UIParam.NX, 128), _p(UIParam.NY, 128), _p(UIParam.NSLICES, 64), _p(UIParam.TR, 500.0)]),
    ),
    Case(
        "fse_2d",
        "256² × 20 slices, ETL 16",
        dict([_p(UIParam.NX, 256), _p(UIParam.NY, 256), _p(UIParam.NSLICES, 20), _p(UIParam.ETL, 16)]),
    ),
    Case(
        "epi_2d",
        "256² × 20 slices, ETL 16",
        dict([_p(UIParam.NX, 256), _p(UIParam.NY, 256), _p(UIParam.NSLICES, 20), _p(UIParam.ETL, 16)]),
    ),
    Case("bssfp_2d", "128² × 1 frame", dict([_p(UIParam.NX, 128), _p(UIParam.NY, 128), _p(UIParam.NUM_FRAMES, 1)])),
    Case(
        "bssfp_2d",
        "128² × 64 frames",
        dict([_p(UIParam.NX, 128), _p(UIParam.NY, 128), _p(UIParam.NUM_FRAMES, 64)]),
    ),
    Case(
        "mprage_3d",
        "192² × 128 par",
        dict([_p(UIParam.NX, 192), _p(UIParam.NY, 192), _p(UIParam.NSLICES, 128)]),
    ),
    Case(
        "gre_noncart_2d",
        "spiral, 48 short shots × 16 slices",
        dict(
            [_p(UIParam.NX, 256), _p(UIParam.NUM_SHOTS, 48), _p(UIParam.NSLICES, 16), _p(UIParam.TR, 160.0)]
        ),
    ),
    Case(
        "gre_noncart_2d",
        "spiral, 4 long shots × 16 slices",
        dict(
            [_p(UIParam.NX, 256), _p(UIParam.NUM_SHOTS, 4), _p(UIParam.NSLICES, 16), _p(UIParam.TR, 800.0)]
        ),
    ),
]


def load_plugin(name: str):
    """Exec-load a zoo plugin by file, the way the bridge does."""
    path = ZOO / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PLUGIN


class _WriteTimer:
    """Stand in for ``pulserver.io.write`` and time its two halves.

    The plugins call ``pio.write(seq, output=path)`` as their last statement,
    so replacing the module attribute is enough to see the call. Serialisation
    and the disk write are timed apart by asking the real ``write`` for the
    payload and storing it here.
    """

    def __init__(self):
        self.serialise_s = 0.0
        self.disk_s = 0.0
        self.payload_bytes = 0
        self._real = pio.write

    def __enter__(self):
        pio.write = self
        return self

    def __exit__(self, *exc):
        pio.write = self._real
        return False

    def __call__(self, seq, output=None, **kwargs):
        t0 = time.perf_counter()
        payload = self._real(seq, output=None, **kwargs)
        self.serialise_s = time.perf_counter() - t0
        self.payload_bytes = len(payload)

        if output is not None:
            t0 = time.perf_counter()
            Path(output).write_bytes(payload)
            self.disk_s = time.perf_counter() - t0
        return None


def run_case(case: Case, out_dir: Path, repeats: int, *, c_only: bool = False) -> dict:
    plugin = load_plugin(case.plugin)
    opts = pp.Opts()
    protocol = plugin.get_default_protocol(opts)
    for key, value in case.overrides.items():
        if key not in protocol:
            raise KeyError(f"{case.plugin}: no protocol entry {key!r}")
        protocol[key]["value"] = value

    verdict = plugin.validate_protocol(opts, protocol)
    if not verdict.get("valid", False):
        return {"plugin": case.plugin, "label": case.label, "error": verdict.get("info", "invalid protocol")}

    seq_path = out_dir / f"{case.plugin}_{case.label.replace(' ', '_').replace('/', '_')}.seq"

    if c_only and seq_path.exists():
        # Re-time the C stages against .seq files a previous run produced. The
        # host-side design timings are the slow part of the sweep and do not
        # change when only the C library is rebuilt.
        return {
            "plugin": case.plugin,
            "label": case.label,
            "duration_s": verdict.get("duration"),
            **_run_c_stages(seq_path, repeats),
        }

    best_validate = min(_time(lambda: plugin.validate_protocol(opts, protocol)) for _ in range(repeats))
    best = None
    for _ in range(repeats):
        gc.collect()
        with _WriteTimer() as timer:
            t0 = time.perf_counter()
            plugin.make_sequence(opts, protocol, str(seq_path))
            total = time.perf_counter() - t0

        row = {
            "design_s": total - timer.serialise_s - timer.disk_s,
            "serialise_s": timer.serialise_s,
            "disk_s": timer.disk_s,
            "payload_bytes": timer.payload_bytes,
        }
        if best is None or row["design_s"] < best["design_s"]:
            best = row

    # Memory in its own pass: tracemalloc hooks every allocation and inflates
    # the very timings it would be measured alongside — several-fold on a
    # sequence this allocation-heavy.
    gc.collect()
    tracemalloc.start()
    with _WriteTimer():
        plugin.make_sequence(opts, protocol, str(seq_path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "plugin": case.plugin,
        "label": case.label,
        "validate_s": best_validate,
        "duration_s": verdict.get("duration"),
        "python_peak_kb": peak / 1024.0,
        **best,
    }

    result.update(_run_c_stages(seq_path, repeats))
    return result


def _run_c_stages(seq_path: Path, repeats: int) -> dict:
    """Time the C pipeline on an existing .seq, from a cold cache."""
    if not BENCH_PIPELINE.exists():
        return {"c_error": f"{BENCH_PIPELINE} not built; run docs/_bench/build_bench.sh"}

    # A stale cache beside the .seq would be loaded instead of re-parsed, and
    # the parse timing would be a lie.
    for stale in (seq_path.with_suffix(".pseg"), seq_path.with_suffix(".pge")):
        stale.unlink(missing_ok=True)

    proc = subprocess.run(
        [str(BENCH_PIPELINE), str(seq_path), str(repeats)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"c_error": proc.stderr.strip()}
    return json.loads(proc.stdout)


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results.json")
    parser.add_argument("--work-dir", type=Path, default=None, help="Where .seq files are written")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default=None, help="Substring filter on plugin name")
    parser.add_argument(
        "--c-only",
        action="store_true",
        help="Re-time the C stages against .seq files already in the work dir",
    )
    args = parser.parse_args(argv)

    work_dir = args.work_dir or (Path(__file__).resolve().parent / "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        if args.only and args.only not in case.plugin:
            continue
        print(f"{case.plugin:<16} {case.label}", flush=True)
        try:
            row = run_case(case, work_dir, args.repeats, c_only=args.c_only)
        except Exception as exc:  # noqa: BLE001 - a broken case must not stop the sweep
            row = {"plugin": case.plugin, "label": case.label, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"    {json.dumps({k: v for k, v in row.items() if k != 'seq'})}", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
