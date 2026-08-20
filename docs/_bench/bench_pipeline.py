"""Stage-by-stage cost of the whole path, from an empty Sequence to a checked scan.

Documentation-only tooling, like everything else in this directory. It is the
harness behind :doc:`../explanations/performance/index` and its subsections,
and it reports exactly the stages those pages are organised into:

``creation``
    The design loop, ``TransformFOV``, the three checks a design script can
    run, deduplication, and serialisation to text and to binary.
``conversion``
    The scanner side of the same file: parse plus conversion, TR detection,
    segmentation, consistency, and the ``.pseg`` cache written and read back.
``safety``
    The gradient checks, PNS and mechanical resonance, each over the scan and
    over the canonical TR, so the window's value is a measured ratio.

The throughput cases are three MPRAGE protocols at one encoding size -- 512
partitions, 1024 views per inversion train -- so the only thing that varies is
how the in-plane shot is held:

``cartesian``
    One phase-encode line per view, the line readout scaled per shot.
``spiral_rotated``
    One spiral arm turned per shot by a ``ROTATIONS`` extension, so the
    waveform is stored once however many arms the scan plays.
``spiral_explicit``
    The same arms written out as their own waveforms, which is the path a
    reader that will not compose a rotation has to take.

The safety sweep is smaller on purpose: a blipped EPI train is where the two
expensive analyses are hardest, and what is being measured there is how each
check responds to scan length rather than a protocol-scale wall clock.

The MPRAGE cases are scale cases sized to measure what the path costs per block
at protocol size, not clinically prescribed protocols.

Run it as::

    python docs/_bench/bench_pipeline.py                    # everything
    python docs/_bench/bench_pipeline.py --stage=creation
    python docs/_bench/bench_pipeline.py --only=cartesian
    python docs/_bench/bench_pipeline.py --scale=0.25       # a quarter-size sweep

Results land in ``docs/_bench/bench_pipeline.json``.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "bench_pipeline.json"

#: The encoding size every throughput case is measured at.
N_Z = 512
VIEWS_PER_TRAIN = 1024

#: In-plane matrix per family, and the interleave count the spiral is designed
#: for. The spiral is held at 128 because its rewinder cannot be rotated above
#: that: the bridge undoes a two-axis k-vector, so both axes run near the
#: gradient ceiling and the vector magnitude is about sqrt(2) times it, which a
#: rotation can land on a single axis.
N_X_CARTESIAN = 512
N_X_SPIRAL = 128
DESIGN_INTERLEAVES = 48

#: A 1024-view train is 15 s (Cartesian) to 29 s (spiral) long, so TI and the
#: outer TR are set to hold it. These are scale cases -- what the path costs
#: per block at protocol size -- not clinically prescribed protocols.
TI = 8.0
TR_OUTER_CARTESIAN = 20.0
TR_OUTER_SPIRAL = 40.0

#: A real offset, so the transform has work to do rather than a no-op to skip.
FOV_OFFSET = (12e-3, -8e-3, 5e-3)

#: Representative Irnich constants -- a generic body-gradient point, not any
#: particular scanner's configuration.
IRNICH = {"chronaxie_us": 360.0, "rheobase": 4.25e8, "alpha": 0.333}

#: Synthetic zero-tolerance forbidden bands (Hz), placed where echo trains
#: actually put energy so the verdict is exercised rather than trivially met.
BANDS = [(550.0, 650.0, 0.0), (1100.0, 1250.0, 0.0)]


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return ``(elapsed_seconds, result)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - start, result


def timed_transform():
    """Patch ``TransformFOV.apply_to_sequence`` to accumulate its own time.

    Returns a one-element list the elapsed seconds land in. The plugin is run
    unmodified, so the transform is measured exactly where it actually happens:
    at the end of ``main``, over the finished scan.
    """
    import pulserver.pypulseq as pp

    elapsed = [0.0]
    original = pp.TransformFOV.apply_to_sequence

    def instrumented(self, seq, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, seq, *args, **kwargs)
        finally:
            elapsed[0] += time.perf_counter() - start

    pp.TransformFOV.apply_to_sequence = instrumented
    return elapsed


def cartesian(n_z: int, views: int):
    """Cartesian MPRAGE: one phase-encode line per view."""
    from pulserver.app import mprage3D_sequence

    return mprage3D_sequence(
        n_x=N_X_CARTESIAN,
        n_y=views,
        n_z=n_z,
        slab_thickness=0.192,
        views_per_segment=views,
        fov_offset=FOV_OFFSET,
        ti=TI,
        tr_outer=TR_OUTER_CARTESIAN,
        n_acs=0,
        n_acs_z=0,
        elliptical=False,
        plot=False,
        write_seq=False,
    )


def spiral(n_z: int, views: int, *, use_rotation_ext: bool):
    """Stack-of-spirals MPRAGE: one arm per view, turned or written out."""
    from pulserver.app import mprage_stack_of_spirals3D_sequence

    return mprage_stack_of_spirals3D_sequence(
        n_x=N_X_SPIRAL,
        n_z=n_z,
        slab_thickness=0.192,
        n_arms=views,
        etl=views,
        design_interleaves=DESIGN_INTERLEAVES,
        fov_offset=FOV_OFFSET,
        ti=TI,
        tr_outer=TR_OUTER_SPIRAL,
        use_rotation_ext=use_rotation_ext,
        plot=False,
        write_seq=False,
    )


CASES = {
    "cartesian": lambda z, v: cartesian(z, v),
    "spiral_rotated": lambda z, v: spiral(z, v, use_rotation_ext=True),
    "spiral_explicit": lambda z, v: spiral(z, v, use_rotation_ext=False),
}


def _collection_args():
    """The system arguments every ``_PulseqCollection`` here is built with."""
    from pulserver.pypulseq import Opts

    system = Opts()
    return (
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


#: How many blocks ``check_timing`` is measured over. It is the one check
#: whose cost is per block -- it materialises an upstream PyPulseq sequence
#: over the window it is given -- so at protocol scale it is reported as a
#: rate measured on a window rather than run over the whole scan.
TIMING_WINDOW_BLOCKS = 20_000


def _timing_over_a_window(seq):
    """``check_timing`` on the first ``TIMING_WINDOW_BLOCKS`` blocks."""
    import numpy as np

    blocks = min(int(seq.num_blocks), TIMING_WINDOW_BLOCKS)
    if blocks == int(seq.num_blocks):
        elapsed, (ok, _) = timed(seq.check_timing)
        return blocks, elapsed, ok
    end = float(np.cumsum(seq.block_durations)[blocks - 1])
    elapsed, (ok, _) = timed(seq.check_timing, time_range=[0.0, end])
    return blocks, elapsed, ok


def creation(name: str, n_z: int, views: int) -> tuple[dict, object, bytes, bytes]:
    """Design, check, deduplicate and serialise one case."""
    transform = timed_transform()
    total, seq = timed(CASES[name], n_z, views)
    design = total - transform[0]
    blocks = seq.num_blocks

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ``write()`` runs both of these before deduplicating, so the
        # amplitude/slew pass sees the library as the design left it. The
        # continuity check is then the first caller to need the C collection
        # and pays for building it, which is the "cold" figure; "warm" is the
        # pass on its own.
        t_limits, (limits_ok, _) = timed(seq.check_hardware_limits)
        t_continuity_cold, (continuity_ok, _) = timed(seq.check_gradient_continuity)
        t_continuity, _ = timed(seq.check_gradient_continuity)
        checked, t_timing, timing_ok = _timing_over_a_window(seq)

    t_dedup, deduped = timed(seq.remove_duplicates)
    t_text, text = timed(deduped._to_text, create_signature=False)
    t_binary, binary = timed(deduped._to_binary)

    entry = {
        "blocks": blocks,
        "design_s": design,
        "transform_fov_s": transform[0],
        "check_hardware_limits_s": t_limits,
        "check_gradient_continuity_s": t_continuity,
        "check_gradient_continuity_cold_s": t_continuity_cold,
        "check_timing_s": t_timing,
        "check_timing_blocks": checked,
        "checks_ok": bool(limits_ok and continuity_ok and timing_ok),
        "remove_duplicates_s": t_dedup,
        "to_text_s": t_text,
        "to_binary_s": t_binary,
        "text_bytes": len(text),
        "binary_bytes": len(binary),
        "shapes": int(deduped._native.num_shapes()),
        "gradients": int(deduped._native.num_gradients()),
        "peak_rss_mb": peak_rss_mb(),
    }
    return entry, deduped, text, binary


def conversion(text: bytes, binary: bytes) -> dict:
    """The scanner side of a written file, stage by stage."""
    from pulserver._ext.pulseg import (
        _PulseqCollection,
        _check_consistency,
        _find_tr,
        _get_segments,
    )

    args = _collection_args()
    t_text, _ = timed(_PulseqCollection, [text], *args)
    t_binary, collection = timed(_PulseqCollection, [binary], *args)
    t_find_tr, tr = timed(_find_tr, collection)
    t_segments, segments = timed(_get_segments, collection)
    t_consistency, _ = timed(_check_consistency, collection)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.seq"
        path.write_bytes(text)
        t_cache_write, _ = timed(_PulseqCollection, str(path), *args)
        cache = path.with_suffix(".pseg")
        cache_bytes = cache.stat().st_size if cache.exists() else 0
        t_cache_read, _ = timed(_PulseqCollection, str(path), *args)

    return {
        "parse_convert_text_s": t_text,
        "parse_convert_binary_s": t_binary,
        "find_tr_s": t_find_tr,
        "segments_s": t_segments,
        "consistency_s": t_consistency,
        "cache_write_s": t_cache_write,
        "cache_read_s": t_cache_read,
        "cache_bytes": cache_bytes,
        "tr_size": int(tr["tr_size"]),
        "num_trs": int(tr["num_trs"]),
        "tr_duration_s": float(tr["tr_duration_us"]) * 1e-6,
        "num_segments": len(segments),
        "peak_rss_mb": peak_rss_mb(),
    }


#: The safety sweep: one family at growing scan length, with the repetition
#: itself held fixed. The point every safety page makes is that the checks
#: cost what the window costs and not what the scan costs, and a sweep is the
#: only way to show that rather than assert it.
SAFETY_SWEEP = [
    {"n_x": 96, "n_y": 96, "n_slices": 1},
    {"n_x": 96, "n_y": 96, "n_slices": 4},
    {"n_x": 96, "n_y": 96, "n_slices": 16},
    {"n_x": 96, "n_y": 96, "n_slices": 48},
]

#: The family the sweep is run on. A blipped echo train is where the two
#: expensive analyses are hardest: the nerve response never returns to
#: baseline between blips, and the echo train is a sharp spectral comb.
SAFETY_FAMILY = "epi2D"


def _build(name: str, **kwargs):
    import pulserver.app as app

    return getattr(app, name + "_sequence")(plot=False, write_seq=False, **kwargs)


def safety(kwargs: dict) -> dict:
    """Every check on one scan, over the whole timeline and over the canonical TR."""
    import numpy as np
    from pypulseq.utils.safe_pns_prediction import safe_example_hw

    seq = _build(SAFETY_FAMILY, **kwargs)
    entry = {
        "kwargs": kwargs,
        "blocks": int(seq.num_blocks),
        "num_trs": int(seq.num_trs),
        "tr_size": int(seq.tr_size),
        "shapes": int(seq._native.num_shapes()),
        "gradients": int(seq._native.num_gradients()),
    }

    seq.declare_tr()
    # Before and after deduplication, because the amplitude/slew pass walks
    # the gradient library and the library only collapses to its distinct
    # rows when the sequence is deduplicated -- which is what writing does.
    t_limits, _ = timed(seq.check_hardware_limits)
    t_continuity, _ = timed(seq.check_gradient_continuity)
    entry["check_hardware_limits_s"] = t_limits
    entry["check_gradient_continuity_s"] = t_continuity

    deduped = seq.remove_duplicates()
    deduped.declare_tr()  # so the timing below is the check, not the structure build
    t_limits_dedup, _ = timed(deduped.check_hardware_limits)
    t_continuity_dedup, _ = timed(deduped.check_gradient_continuity)
    entry["gradients_deduplicated"] = int(deduped._native.num_gradients())
    entry["check_hardware_limits_deduplicated_s"] = t_limits_dedup
    entry["check_gradient_continuity_deduplicated_s"] = t_continuity_dedup

    # SAFE on both sides of the PNS comparison, so the only thing that varies
    # is the window: upstream's timeline against the canonical TR the gate
    # judges. Irnich is timed alongside because it is the model the
    # scanner-side gate actually applies.
    safe_hw = safe_example_hw()
    t_pns_timeline, timeline = timed(seq.calculate_pns, safe_hw, do_plots=False, tr=None)
    t_pns_safe_tr, safe_tr = timed(
        seq.calculate_pns, safe_hw, do_plots=False, tr="worst_case"
    )
    t_pns_irnich_tr, _ = timed(seq.calculate_pns, IRNICH, do_plots=False, tr="worst_case")
    entry["pns_timeline_safe_s"] = t_pns_timeline
    entry["pns_worst_case_tr_safe_s"] = t_pns_safe_tr
    entry["pns_worst_case_tr_irnich_s"] = t_pns_irnich_tr
    entry["pns_timeline_safe_peak"] = float(np.max(timeline[1]))
    entry["pns_worst_case_tr_safe_peak"] = float(np.max(safe_tr[1]))

    t_spectrum_timeline, _ = timed(seq.calculate_gradient_spectrum, plot=False, tr=None)
    t_gate, gate = timed(
        seq.calculate_gradient_spectrum,
        plot=False,
        tr="worst_case",
        resonance_lines=True,
        bands=BANDS,
    )
    t_comb, _ = timed(
        seq.calculate_gradient_spectrum, plot=False, tr="worst_case", resonance_lines=False
    )
    entry["mechres_timeline_s"] = t_spectrum_timeline
    entry["mechres_gate_s"] = t_gate
    entry["mechres_comb_s"] = t_comb
    entry["mechres_ok"] = bool(gate[-1].ok)
    entry["peak_rss_mb"] = peak_rss_mb()
    return entry


def report_creation(name: str, entry: dict) -> None:
    blocks = entry["blocks"]
    print(
        f"  {name:<16s} blocks {blocks:>9d}"
        f"  design {entry['design_s']:7.2f}s ({entry['design_s'] / blocks * 1e6:5.2f} us/block)"
        f"  fov {entry['transform_fov_s']:6.2f}s"
        f"  dedup {entry['remove_duplicates_s']:6.2f}s"
        f"  text {entry['to_text_s']:6.2f}s  binary {entry['to_binary_s']:6.2f}s"
        f"  shapes {entry['shapes']:>6d}  peak RSS {entry['peak_rss_mb']:7.0f} MB",
        flush=True,
    )
    print(
        f"  {name:<16s} checks: limits {entry['check_hardware_limits_s'] * 1e3:8.1f} ms"
        f"  continuity {entry['check_gradient_continuity_s'] * 1e3:8.1f} ms"
        f" (cold {entry['check_gradient_continuity_cold_s'] * 1e3:8.1f} ms)"
        f"  timing {entry['check_timing_s'] * 1e3:8.1f} ms"
        f" over {entry['check_timing_blocks']} blocks"
        f" ({entry['check_timing_s'] / entry['check_timing_blocks'] * 1e6:5.1f} us/block)",
        flush=True,
    )


def report_conversion(name: str, entry: dict) -> None:
    print(
        f"  {name:<16s} parse+convert text {entry['parse_convert_text_s'] * 1e3:8.1f} ms"
        f"  binary {entry['parse_convert_binary_s'] * 1e3:8.1f} ms"
        f"  find_tr {entry['find_tr_s'] * 1e3:7.2f} ms"
        f"  segments {entry['segments_s'] * 1e3:7.2f} ms"
        f"  consistency {entry['consistency_s'] * 1e3:7.2f} ms",
        flush=True,
    )
    print(
        f"  {name:<16s} cache write {entry['cache_write_s'] * 1e3:8.1f} ms"
        f"  cache read {entry['cache_read_s'] * 1e3:8.1f} ms"
        f"  ({entry['cache_bytes'] / 1e6:.1f} MB)"
        f"  tr_size {entry['tr_size']}  num_trs {entry['num_trs']}"
        f"  segments {entry['num_segments']}",
        flush=True,
    )


def report_safety(entry: dict) -> None:
    label = f"{entry['blocks']} blk"
    print(
        f"  {label:<12s} shapes {entry['shapes']:>4d}  num_trs {entry['num_trs']:>5d}"
        f"  grads {entry['gradients']:>6d}->{entry['gradients_deduplicated']:<5d}"
        f"  limits {entry['check_hardware_limits_s'] * 1e3:7.1f}"
        f" -> {entry['check_hardware_limits_deduplicated_s'] * 1e3:6.1f} ms"
        f"  continuity {entry['check_gradient_continuity_s'] * 1e3:6.1f}"
        f" -> {entry['check_gradient_continuity_deduplicated_s'] * 1e3:6.1f} ms",
        flush=True,
    )
    print(
        f"  {label:<12s} PNS timeline {entry['pns_timeline_safe_s'] * 1e3:9.1f} ms"
        f"  TR(SAFE) {entry['pns_worst_case_tr_safe_s'] * 1e3:7.1f} ms"
        f"  TR(Irnich) {entry['pns_worst_case_tr_irnich_s'] * 1e3:7.1f} ms",
        flush=True,
    )
    print(
        f"  {label:<12s} spectrum timeline {entry['mechres_timeline_s'] * 1e3:9.1f} ms"
        f"  gate {entry['mechres_gate_s'] * 1e3:7.1f} ms"
        f"  comb {entry['mechres_comb_s'] * 1e3:7.1f} ms"
        f"  ok {entry['mechres_ok']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(CASES), help="run one throughput case")
    parser.add_argument(
        "--stage",
        choices=["creation", "conversion", "safety"],
        action="append",
        help="run one stage (repeatable); default is all three",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="shrink the encoding size")
    args = parser.parse_args()

    stages = args.stage or ["creation", "conversion", "safety"]
    n_z = max(2, int(N_Z * args.scale))
    views = max(2, int(VIEWS_PER_TRAIN * args.scale))

    results: dict = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text())
    results.setdefault("creation", {})
    results.setdefault("conversion", {})
    results.setdefault("safety", {})

    if "creation" in stages or "conversion" in stages:
        print(f"throughput cases ({n_z} partitions x {views} views)", flush=True)
        for name in [args.only] if args.only else sorted(CASES):
            entry, _, text, binary = creation(name, n_z, views)
            entry["n_z"] = n_z
            entry["views"] = views
            results["creation"][name] = entry
            report_creation(name, entry)
            if "conversion" in stages:
                converted = conversion(text, binary)
                converted["blocks"] = entry["blocks"]
                results["conversion"][name] = converted
                report_conversion(name, converted)
            RESULTS.write_text(json.dumps(results, indent=2))

    if "safety" in stages:
        print(f"safety sweep ({SAFETY_FAMILY}, scan length varied, repetition fixed)", flush=True)
        results["safety"] = {"family": SAFETY_FAMILY, "sweep": []}
        for kwargs in SAFETY_SWEEP:
            entry = safety(kwargs)
            results["safety"]["sweep"].append(entry)
            report_safety(entry)
            RESULTS.write_text(json.dumps(results, indent=2))

    print(f"results -> {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
