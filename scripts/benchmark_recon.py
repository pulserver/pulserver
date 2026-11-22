"""Time and peak memory of each shipped reconstruction plugin over a series the virtual scanner acquires.

Each case is scanned once with ``pulserver scan`` (kept under ``--work``),
then replayed with ``pulserver replay`` in a fresh process per case and per
repeat, so the peak resident memory of the process is that of the
reconstruction alone. The table is written as JSON and, with ``--figure``,
drawn as bars of wall time and peak memory per case, and of the ratio between
two runs with ``--compare``.

Examples
--------
::

    python scripts/benchmark_recon.py --work /tmp/bench --out before.json
    python scripts/benchmark_recon.py --work /tmp/bench --out after.json \
        --compare before.json --figure bench.png
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "src" / "pulserver" / "_zoo" / "recon"

#: Limits each logical axis is designed under, so any orientation stays within them.
LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
    "B0": 3.0,
    "design_max_grad": 40.0 / 3**0.5,
    "design_max_slew": 150.0 / 3**0.5,
}

#: ``name: (sequence plugin, reconstruction plugin, protocol, coil)``.
CASES = {
    "gre2d": ("gre2d", "pics", {"nx": 192, "ny": 192, "nslices": 8}, "head8/head32"),
    "gre2d-R2": (
        "gre2d",
        "pics",
        {"nx": 192, "ny": 192, "nslices": 8, "Ry": 2},
        "head8/head32",
    ),
    "gre3d": ("gre3d", "pics", {"nx": 128, "ny": 128, "nslices": 32}, "head8/head32"),
    "radial2d": ("gre_radial2d", "nufft", {"nx": 192, "nslices": 8}, "head8/head32"),
    "spiral2d": (
        "gre_spiral2d",
        "nufft",
        {"nx": 192, "nslices": 8, "num_shots": 16},
        "head8/head32",
    ),
    "propeller2d": (
        "gre_propeller2d",
        "nufft",
        {"nx": 192, "nslices": 8},
        "head8/head32",
    ),
    "stars3d": (
        "gre_stack_of_stars3d",
        "nufft",
        {"nx": 128, "nslices": 32},
        "head8/head32",
    ),
    "epi2d-mb2-fmri": (
        "epi2d",
        "epi",
        {"nx": 96, "ny": 96, "nslices": 12, "num_frames": 10, "multiband": 2},
        "head8/head32",
    ),
    "epi2d-fmri": (
        "epi2d",
        "epi",
        {"nx": 96, "ny": 96, "nslices": 12, "num_frames": 10, "Ry": 2},
        "head8/head32",
    ),
}


def scan(name: str, work: Path) -> tuple[Path, Path]:
    """Scan a case once; return its MRD file and design store."""
    plugin, _, protocol, coil = CASES[name]
    directory = work / name
    raw, store = directory / "raw.h5", directory / "designs"
    if raw.is_file():
        return raw, store
    directory.mkdir(parents=True, exist_ok=True)
    from pulserver.host._blocks import format_limits
    from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

    (directory / "limits.txt").write_text(format_limits(LIMITS))

    lines = "\n".join(f"{key}: {value}" for key, value in protocol.items())
    (directory / "protocol.txt").write_text(
        f"{PROTOCOL_BEGIN}\n{lines}\n{PROTOCOL_END}\n"
    )
    command = [
        sys.executable,
        "-c",
        "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))",
        "scan",
        "--plugins",
        str(ROOT / "src" / "pulserver" / "_zoo" / "sequences"),
        "--plugin",
        plugin,
        "--protocol",
        str(directory / "protocol.txt"),
        "--limits",
        str(directory / "limits.txt"),
        "--store",
        str(store),
        "--mrd",
        str(raw),
        "--coil",
        coil,
        "--phantom",
        "brainweb",
        "--spacing",
        "2",
    ]
    subprocess.run(command, check=True, cwd=directory)  # noqa: S603 -- this script's own command
    return raw, store


def replay(name: str, raw: Path, store: Path, output: Path) -> dict[str, float]:
    """Replay a case in a fresh process; return its wall time (s) and peak resident memory (GiB)."""
    output.mkdir(parents=True, exist_ok=True)
    plugin = RECON / f"{CASES[name][1]}.py"
    command = [
        sys.executable,
        "-c",
        "import resource, sys, time\n"
        "try:\n"
        "    from pulserver._threads import serial_blas\n"
        "    serial_blas()\n"
        "except ImportError:\n"
        "    pass\n"
        "import ismrmrd\n"
        "from pulserver.recon._replay import main\n"
        "from pulserver.recon._runtime.offline import _FileConnection\n"
        "import pulserver.recon.handlers.pics, bartorch\n"
        "start = time.perf_counter()\n"
        "dataset = ismrmrd.Dataset(sys.argv[1], 'dataset', create_if_needed=False)\n"
        "for _ in _FileConnection(dataset)._items(): pass\n"
        "dataset.close()\n"
        "read = time.perf_counter() - start\n"
        "start = time.perf_counter()\n"
        "status = main(sys.argv[1:])\n"
        "elapsed = time.perf_counter() - start\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20\n"
        "print(f'BENCH {elapsed - read:.3f} {read:.3f} {peak:.3f}')\n"
        "sys.exit(status)\n",
        str(raw),
        "--plugin",
        str(plugin),
        "--store",
        str(store),
        "--output",
        str(output),
    ]
    start = time.perf_counter()
    run = subprocess.run(  # noqa: S603 -- this script's own command
        command, capture_output=True, text=True, env=os.environ
    )
    wall = time.perf_counter() - start
    if run.returncode != 0:
        raise RuntimeError(f"{name} replay failed:\n{run.stderr[-4000:]}")
    line = next(line for line in run.stdout.splitlines() if line.startswith("BENCH"))
    _, recon_s, read_s, peak_gib = line.split()
    return {
        "recon_s": float(recon_s),
        "read_s": float(read_s),
        "process_s": wall,
        "peak_gib": float(peak_gib),
    }


def figure(results: dict, baseline: dict | None, path: Path) -> None:
    """Draw wall time and peak memory per case, beside the baseline's where given."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    x = np.arange(len(names))
    panels = (
        ("recon_s", "Reconstruction time (s)"),
        ("peak_gib", "Peak host memory (GiB)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    width = 0.38 if baseline else 0.6
    for ax, (key, label) in zip(axes, panels, strict=True):
        after = [results[n][key] for n in names]
        if baseline:
            before = [baseline.get(n, {}).get(key, np.nan) for n in names]
            ax.bar(x - width / 2, before, width, label="before", color="#9aa4b1")
            bars = ax.bar(x + width / 2, after, width, label="after", color="#2f6fd6")
            for bar, b, a in zip(bars, before, after, strict=True):
                if np.isfinite(b) and a > 0:
                    ax.annotate(
                        f"{b / a:.1f}x",
                        (bar.get_x() + bar.get_width() / 2, a),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.legend(frameon=False)
        else:
            ax.bar(x, after, width, color="#2f6fd6")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    results = {}
    for name in args.cases:
        raw, store = scan(name, args.work)
        runs = [
            replay(
                name, raw, store, args.work / name / f"images-{args.out.stem}-{repeat}"
            )
            for repeat in range(args.repeats)
        ]
        results[name] = {key: min(run[key] for run in runs) for key in runs[0]}
        print(name, results[name], flush=True)
    if args.out.is_file():
        results = {**json.loads(args.out.read_text()), **results}
    args.out.write_text(json.dumps(results, indent=2))
    if args.figure is not None:
        baseline = json.loads(args.compare.read_text()) if args.compare else None
        figure(results, baseline, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
