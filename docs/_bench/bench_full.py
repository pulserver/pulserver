"""Every shipped plugin, at four problem sizes, on the two clocks an operator feels.

Documentation-only tooling, like everything else in this directory. It is the
harness behind :doc:`../explanations/performance/full_benchmark`, and unlike the
other benchmarks here it measures the plugins *through the scanner protocol
contract* rather than through ``main``: a size is a protocol the console could
have prescribed, and the two entry points timed are the two the console calls.

``validate``
    ``SequencePlugin.validate_protocol`` -- what runs on every parameter the
    operator touches, so it decides whether the UI feels immediate. Reported as
    the median of repeated calls in a warm process, which is the state a
    scanner-side plugin server is in.

``save_rx``
    Everything one press of *Save Rx* costs, end to end and in one process:
    ``SequencePlugin.make_sequence`` (the design loop plus deduplication plus
    the binary write), then the interpreter's ``pulseg_read`` -- parse,
    conversion and the binary cache written beside the file -- then
    ``pulseg_check_safety`` over the canonical TR. Its peak resident set size
    is the memory the scanner host has to find.

Each measurement runs in its own subprocess, so a peak RSS is that case's and
not the high-water mark of the whole sweep.

Run it as::

    python docs/_bench/bench_full.py                 # the whole sweep
    python docs/_bench/bench_full.py --only=gre2D    # one family
    python docs/_bench/bench_full.py --figures-only  # redraw from the JSON

Results land in ``docs/_bench/bench_full.json`` and the figures in
``docs/explanations/assets/full_benchmark/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
ASSETS = DOCS / "explanations" / "assets" / "full_benchmark"
RESULTS = Path(__file__).resolve().parent / "bench_full.json"

#: How many timed ``validate_protocol`` calls are made, after one untimed call
#: to warm the design modules. The fastest is reported: every source of noise
#: on a shared machine adds time, so the minimum is the closest estimate of
#: what the call costs, and the spread is kept beside it.
VALIDATE_REPEATS = 7

#: Representative Irnich constants -- a generic body-gradient point, not any
#: particular scanner's configuration. ``alpha`` is folded into the threshold
#: the safety entry point takes.
IRNICH_CHRONAXIE_US = 360.0
IRNICH_RHEOBASE = 4.25e8
IRNICH_ALPHA = 0.333

#: Two forbidden bands where echo trains and balanced readouts actually put
#: energy, with the amplitude limit left wide open. The gate is being timed,
#: not exercised for verdicts: a band table decides how much spectral work
#: ``pulseg_check_safety`` does -- the resolution and the analysis range come
#: from it -- while a limit no sequence can reach keeps every case running the
#: whole check instead of returning early on the first violation.
BANDS = [(550.0, 650.0, 1e12), (1100.0, 1250.0, 1e12)]

#: A PNS ceiling no case reaches, for the same reason.
PNS_THRESHOLD_PERCENT = 1e6

#: Four protocols per family, as overrides on the plugin's own default
#: protocol. Keys are canonical protocol keys, so a size here is a prescription
#: the console could have sent: ``nslices`` is the partition count on the 3D
#: families, and a ``userN_value`` is that plugin's own control -- arms, spokes,
#: blades, echo train length -- named in its ``get_default_protocol``.
SIZES = {
    "gre2D": [
        {"nx": 64, "ny": 64, "nslices": 1},
        {"nx": 128, "ny": 128, "nslices": 4},
        {"nx": 128, "ny": 128, "nslices": 8},
        {"nx": 256, "ny": 256, "nslices": 16},
    ],
    "gre3D": [
        {"nx": 64, "ny": 64, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 32},
        {"nx": 128, "ny": 128, "nslices": 48},
        {"nx": 256, "ny": 256, "nslices": 64},
    ],
    "se2D": [
        {"nx": 64, "ny": 64, "nslices": 1},
        {"nx": 128, "ny": 128, "nslices": 2},
        {"nx": 128, "ny": 128, "nslices": 4},
        {"nx": 256, "ny": 256, "nslices": 8},
    ],
    "se3D": [
        {"nx": 64, "ny": 64, "nslices": 8},
        {"nx": 128, "ny": 128, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 24},
        {"nx": 128, "ny": 128, "nslices": 32},
    ],
    "fse2D": [
        {"nx": 128, "ny": 128, "nslices": 1},
        {"nx": 128, "ny": 128, "nslices": 2},
        {"nx": 128, "ny": 128, "nslices": 4},
        {"nx": 256, "ny": 256, "nslices": 8},
    ],
    "fse3D": [
        {"nx": 128, "ny": 128, "nslices": 8},
        {"nx": 128, "ny": 128, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 32},
        {"nx": 128, "ny": 128, "nslices": 48},
    ],
    "epi2D": [
        {"nx": 64, "ny": 64, "nslices": 1},
        {"nx": 96, "ny": 96, "nslices": 8},
        {"nx": 128, "ny": 128, "nslices": 12},
        {"nx": 128, "ny": 128, "nslices": 16},
    ],
    "epi3D": [
        {"nx": 64, "ny": 64, "nslices": 8},
        {"nx": 96, "ny": 96, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 24},
        {"nx": 128, "ny": 128, "nslices": 32},
    ],
    "bssfp2D": [
        {"nx": 128, "ny": 128, "nslices": 1},
        {"nx": 128, "ny": 128, "nslices": 2},
        {"nx": 128, "ny": 128, "nslices": 4},
        {"nx": 256, "ny": 256, "nslices": 8},
    ],
    "bssfp3D": [
        {"nx": 96, "ny": 96, "nslices": 8},
        {"nx": 128, "ny": 128, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 32},
        {"nx": 128, "ny": 128, "nslices": 48},
    ],
    "gre_multiecho2D": [
        {"nx": 64, "ny": 64, "nslices": 1},
        {"nx": 128, "ny": 128, "nslices": 2},
        {"nx": 128, "ny": 128, "nslices": 4},
        {"nx": 256, "ny": 256, "nslices": 8},
    ],
    "gre_multiecho3D": [
        {"nx": 64, "ny": 64, "nslices": 8},
        {"nx": 128, "ny": 128, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 32},
        {"nx": 128, "ny": 128, "nslices": 48},
    ],
    "mprage3D": [
        {"nx": 64, "ny": 64, "nslices": 16},
        {"nx": 128, "ny": 128, "nslices": 32},
        {"nx": 128, "ny": 128, "nslices": 64},
        {"nx": 192, "ny": 192, "nslices": 128},
    ],
    "mprage_stack_of_spirals3D": [
        {"nx": 64, "nslices": 16, "user0_value": 32.0},
        {"nx": 128, "nslices": 32, "user0_value": 64.0},
        {"nx": 128, "nslices": 48, "user0_value": 128.0},
        {"nx": 128, "nslices": 64, "user0_value": 128.0},
    ],
    "gre_radial2D": [
        {"nx": 64, "nslices": 1, "user0_value": 101.0},
        {"nx": 128, "nslices": 4, "user0_value": 201.0},
        {"nx": 128, "nslices": 8, "user0_value": 201.0},
        {"nx": 256, "nslices": 8, "user0_value": 403.0},
    ],
    "gre_spiral2D": [
        {"nx": 64, "nslices": 1, "user0_value": 8.0},
        {"nx": 128, "nslices": 4, "user0_value": 16.0},
        {"nx": 128, "nslices": 8, "user0_value": 16.0},
        {"nx": 256, "nslices": 8, "user0_value": 32.0},
    ],
    "gre_stack_of_spirals3D": [
        {"nx": 64, "nslices": 8, "user0_value": 8.0},
        {"nx": 128, "nslices": 16, "user0_value": 16.0},
        {"nx": 128, "nslices": 32, "user0_value": 16.0},
        {"nx": 128, "nslices": 48, "user0_value": 32.0},
    ],
    "gre_stack_of_stars3D": [
        {"nx": 64, "nslices": 8, "user0_value": 101.0},
        {"nx": 128, "nslices": 16, "user0_value": 201.0},
        {"nx": 128, "nslices": 24, "user0_value": 201.0},
        {"nx": 128, "nslices": 32, "user0_value": 403.0},
    ],
    "se_propeller2D": [
        {"nx": 64, "nslices": 1, "user0_value": 16.0, "user1_value": 8.0},
        {"nx": 128, "nslices": 2, "user0_value": 24.0, "user1_value": 12.0},
        {"nx": 128, "nslices": 4, "user0_value": 24.0, "user1_value": 12.0},
        {"nx": 256, "nslices": 4, "user0_value": 32.0, "user1_value": 16.0, "TE": 120.0},
    ],
    "zte3D": [
        {"nx": 32},
        {"nx": 48},
        {"nx": 56},
        {"nx": 64},
    ],
}


# ======================================================================
# The worker: one family at one size, in a process of its own
# ======================================================================


def measure(family: str, index: int) -> dict:
    """Measure one protocol, in this process, and return the record."""
    import importlib
    import resource
    import tempfile
    import time

    import pulserver.pypulseq as pp
    from pulserver import set_protocol_value
    from pulserver._ext.pulseg import _PulseqCollection, _check_safety

    system = pp.Opts()
    module = importlib.import_module(f"pulserver.app.sequence.{family}_sequence")
    plugin = module.PLUGIN

    protocol = plugin.get_default_protocol(system)
    for key, value in SIZES[family][index].items():
        set_protocol_value(protocol, key, value)

    baseline_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # One untimed call first: a scanner-side plugin server has already designed
    # this family at least once by the time an operator is turning knobs.
    report = plugin.validate_protocol(system, protocol)
    validate_ms = []
    for _ in range(VALIDATE_REPEATS):
        start = time.perf_counter()
        plugin.validate_protocol(system, protocol)
        validate_ms.append((time.perf_counter() - start) * 1e3)
    validate_ms.sort()

    # ``num_blocks`` of the scan as written -- the only place it is still in
    # hand, since ``make_sequence`` owns the sequence and does not return it.
    # A plugin may write a linked collection rather than one file, and the
    # scan is then all of them.
    blocks = [0]
    write_binary = pp.Sequence.write_binary

    def counting_write_binary(self, *args, **kwargs):
        blocks[0] += int(self.num_blocks)
        return write_binary(self, *args, **kwargs)

    pp.Sequence.write_binary = counting_write_binary

    collection_args = (
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
        1,
        [],
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.seq"

        start = time.perf_counter()
        plugin.make_sequence(system, protocol, str(path))
        t_make = time.perf_counter() - start

        start = time.perf_counter()
        collection = _PulseqCollection(str(path), *collection_args)
        t_convert = time.perf_counter() - start

        start = time.perf_counter()
        _check_safety(
            collection,
            BANDS,
            IRNICH_RHEOBASE / IRNICH_ALPHA,
            IRNICH_CHRONAXIE_US,
            PNS_THRESHOLD_PERCENT,
            False,
        )
        t_safety = time.perf_counter() - start

        seq_bytes = sum(f.stat().st_size for f in Path(tmp).glob("*.seq"))
        cache_bytes = sum(f.stat().st_size for f in Path(tmp).glob("*.pseg"))

    return {
        "protocol": SIZES[family][index],
        "blocks": blocks[0],
        "duration_s": report.get("duration"),
        "validate_ms": validate_ms[0],
        "validate_ms_median": validate_ms[len(validate_ms) // 2],
        "validate_ms_all": validate_ms,
        "make_sequence_s": t_make,
        "convert_cache_s": t_convert,
        "safety_s": t_safety,
        "save_rx_s": t_make + t_convert + t_safety,
        "seq_bytes": seq_bytes,
        "cache_bytes": cache_bytes,
        "baseline_rss_mb": baseline_rss_mb,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


# ======================================================================
# The sweep
# ======================================================================

#: What separates the worker's JSON from anything the plugin printed.
MARKER = "@@BENCH_FULL@@"


def run_case(family: str, index: int) -> dict:
    """Run one case in a subprocess and return its record."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", family, str(index)],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": " | ".join(tail) or f"exit {proc.returncode}"}


def sweep(names: list[str]) -> dict:
    results = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text())
    for family in names:
        print(family, flush=True)
        entries = []
        for index in range(len(SIZES[family])):
            entry = run_case(family, index)
            entries.append(entry)
            if "error" in entry:
                print(f"  size {index}  FAILED: {entry['error']}", flush=True)
                continue
            print(
                f"  {entry['blocks']:>8d} blocks"
                f"  validate {entry['validate_ms']:7.1f} ms"
                f"  save Rx {entry['save_rx_s']:6.2f} s"
                f"  (design {entry['make_sequence_s']:5.2f}"
                f" + convert {entry['convert_cache_s']:5.2f}"
                f" + safety {entry['safety_s']:5.2f})"
                f"  RSS {entry['peak_rss_mb']:6.0f} MB"
                f"  seq {entry['seq_bytes'] / 1e6:6.1f} MB"
                f"  cache {entry['cache_bytes'] / 1e6:6.1f} MB",
                flush=True,
            )
        results[family] = entries
        RESULTS.write_text(json.dumps(results, indent=2))
    return results


# ======================================================================
# The figures
# ======================================================================

#: One scatter per quantity: (file stem, record field, scale onto the axis
#: unit, axis label, y scale). A logarithmic y is for the quantities that span
#: decades; peak RSS does not, and a linear axis shows what it does do.
PANELS = [
    ("validate", "validate_ms", 1.0, "validate_protocol() [ms]", "log"),
    ("save_rx", "save_rx_s", 1.0, "design + convert + safety + cache [s]", "log"),
    ("peak_rss", "peak_rss_mb", 1.0, "peak resident set size [MB]", "linear"),
    ("seq_size", "seq_bytes", 1e-6, "binary sequence file [MB]", "log"),
    ("cache_size", "cache_bytes", 1e-6, "interpreter cache [MB]", "log"),
]


def figures(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    families = [f for f in SIZES if f in results]
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, 20))
    markers = ["o", "s", "^", "D", "v"]
    ASSETS.mkdir(parents=True, exist_ok=True)

    for stem, field, scale, label, yscale in PANELS:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for i, family in enumerate(families):
            points = [e for e in results[family] if "error" not in e]
            if not points:
                continue
            x = np.array([e["blocks"] for e in points], dtype=float)
            y = np.array([e[field] for e in points], dtype=float) * scale
            color = colors[i % 20]
            ax.scatter(
                x,
                y,
                s=26,
                color=color,
                marker=markers[i % len(markers)],
                label=family,
                zorder=3,
            )
            # The guide is fitted in the space the axes are drawn in, so a
            # straight line means what it looks like it means: a power law
            # where both axes are logarithmic, a proportionality where the
            # y axis is linear.
            if len(x) >= 2 and np.ptp(x) > 0:
                span = np.array([x.min(), x.max()])
                if yscale == "log" and y.min() > 0:
                    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
                    guide = np.exp(intercept) * span**slope
                else:
                    slope, intercept = np.polyfit(np.log(x), y, 1)
                    guide = slope * np.log(span) + intercept
                ax.plot(span, guide, color=color, lw=1.0, alpha=0.55)

        ax.set_xscale("log")
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("sequence size [blocks]")
        ax.set_ylabel(label)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(
            fontsize=7,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout()
        fig.savefig(ASSETS / f"{stem}.png", dpi=140)
        plt.close(fig)
        print(f"  {ASSETS / (stem + '.png')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one family")
    parser.add_argument("--case", nargs=2, metavar=("FAMILY", "INDEX"), help=argparse.SUPPRESS)
    parser.add_argument("--figures-only", action="store_true", help="redraw from the JSON")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.case:
        family, index = args.case[0], int(args.case[1])
        print(MARKER + json.dumps(measure(family, index)))
        return 0

    if args.figures_only:
        figures(json.loads(RESULTS.read_text()))
        return 0

    results = sweep([args.only] if args.only else list(SIZES))
    if not args.no_figures:
        figures(results)
    print(f"results -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
