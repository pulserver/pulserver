"""Design-phase cost and memory at the edge of each sequence family.

Documentation-only tooling, like everything else in this directory.

``bench_zoo.py`` measures *representative* protocols — the sizes a clinical
protocol actually uses — across the whole pipeline. This measures the opposite
end: the largest protocol each family admits, where the host-side design pass
is the only stage that matters and where the numbers stop being linear in
anything convenient. A 512 x 1024 x 512 MPRAGE at ETL 1024 is millions of
blocks; the cost per block, and the transient memory the range path holds while
building them, are what decide whether it is feasible at all.

Three things are reported that ``bench_zoo`` does not separate:

===================  ======================================================
``us/block``         design seconds per emitted block. Flat across sizes is
                     the property being defended: anything super-linear
                     shows up here long before the wall time looks wrong.
``rss_delta_mb``     peak resident growth over the pass. The range path
                     buffers each chunk's payload rows as Python tuples
                     before converting them, so this is where a chunk size
                     that is too generous would show.
``events``           rows in the libraries a *block row* points at, once the
                     design pass ends and before ``remove_duplicates`` runs.
                     A range collapses these as it registers them, so a plugin
                     on the range path reports its event *vocabulary* -- a few
                     thousand -- and one still looping over ``add_block``
                     reports roughly one row per event in the scan.
``ext rows``         rows in the libraries an *extension chain* points at, plus
                     the chain nodes. These are never collapsed early (it would
                     reorder chains and change the file), so this is the floor
                     on what a labelled scan costs to build.
===================  ======================================================

There are two families of case, and they answer different questions:

``protocol``
    A zoo plugin at the largest protocol it will actually validate. This is
    the honest ceiling for a scanner operator: at the shipped 220 mm FOV a
    256-point readout already needs a prewinder wider than its own block, so
    the Cartesian families stop well short of what the design path can build.

``module``
    The design modules driven straight, past the protocol caps, at the sizes
    the *writer* has to survive rather than the ones the UI offers -- a
    512 x 1024 x 512 MPRAGE at ETL 1024, echo trains at 128 and 256, EPI and
    bSSFP and spiral and ZTE at pushed resolution. The chronology of each
    mirrors its plugin; only the numbers are bigger.

Run it as::

    python docs/_bench/bench_scale.py                     # every case
    python docs/_bench/bench_scale.py --only=mprage_3d
    python docs/_bench/bench_scale.py --family=module
    python docs/_bench/bench_scale.py --scale=0.5         # half-size sweep
    python docs/_bench/bench_scale.py --out docs/_bench/scale_results.json

``--scale`` multiplies every matrix dimension, so the same case can be run at
several sizes to check that ``us/block`` really is flat. Sizes are chosen to
fit in roughly 16 GB at ``--scale=1``; go higher only with the RAM to match.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
from bench_zoo import load_plugin
from pulserver import UIParam

#: System limits the module-driven cases design against. The plugins derate a
#: shared ``Opts`` in place, so these cases carry their own.
SYSTEM_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}

#: Libraries a *block row* points at, by column. ``add_range`` collapses these
#: as it registers them, so their size is the sequence's event vocabulary.
_EVENT_LIBRARIES = (
    "shape_library", "rf_library", "arb_library", "trap_library", "grad_library", "adc_library",
)

#: Libraries an *extension chain* points at, plus the chain nodes themselves.
#: These are deliberately left one entry per event -- a block orders its chain
#: by reference ID, so collapsing them would reorder chains and change the file
#: -- and they are consequently the floor on what a large scan costs to build.
_EXTENSION_LIBRARIES = (
    "trigger_library", "label_set_library", "label_inc_library",
    "rotation_library", "rf_shim_library", "extensions_library",
)


def _p(key, value):
    return (str(key), value)


@dataclass
class Case:
    """One plugin at the largest protocol its family is expected to survive."""

    plugin: str
    label: str
    overrides: dict = field(default_factory=dict)
    #: Protocol keys that scale with ``--scale``; the rest are held fixed.
    scaled: tuple[str, ...] = ()
    family: str = "protocol"


@dataclass
class ModuleCase:
    """One design module chronology, driven past what any protocol validates.

    ``build(scale)`` returns ``(label, callable)``; calling that callable
    designs the sequence and returns it. The split matters: everything the case
    can hoist out of the timed region -- trajectory design, sampling patterns,
    the modules themselves -- belongs in ``build``, because this benchmark is
    about registering blocks, not about designing one readout.
    """

    plugin: str
    label: str
    build: Any
    family: str = "module"


#: The edge of each family, not the middle. Every case is chosen so the design
#: pass dominates: long echo trains (one shot covers many blocks), or many
#: shots, or both.
CASES = [
    # The Cartesian cases stop where the *protocol* stops, not where the design
    # path does: at the shipped 220 mm FOV a 256-point readout already needs a
    # prewinder wider than its own block, so these are the largest sizes the
    # plugins actually validate. Drive the modules directly if you want a
    # 512 x 1024 x 512 shot count -- the sequence writer does not care.
    Case(
        "mprage_3d",
        "192 x 192 x 192 (protocol max)",
        dict([_p(UIParam.NX, 192), _p(UIParam.NY, 192), _p(UIParam.NSLICES, 192)]),
        scaled=(str(UIParam.NY), str(UIParam.NSLICES)),
    ),
    Case(
        "gre_3d",
        "128 x 256 x 256 (protocol max)",
        dict([_p(UIParam.NX, 128), _p(UIParam.NY, 256), _p(UIParam.NSLICES, 256), _p(UIParam.TR, 2000.0)]),
        scaled=(str(UIParam.NY), str(UIParam.NSLICES)),
    ),
    Case(
        "fse_3d",
        "256 x 256 x 128, ETL 128",
        dict(
            [
                _p(UIParam.NX, 256), _p(UIParam.NY, 256), _p(UIParam.NSLICES, 128),
                _p(UIParam.ETL, 128), _p(UIParam.TR, 4000.0),
            ]
        ),
        scaled=(str(UIParam.NY), str(UIParam.NSLICES)),
    ),
    Case(
        "epi_3d",
        "128 x 128 x 128, ETL 128",
        dict(
            [
                _p(UIParam.NX, 128), _p(UIParam.NY, 128), _p(UIParam.NSLICES, 128),
                _p(UIParam.ETL, 128), _p(UIParam.TR, 4000.0),
            ]
        ),
        scaled=(str(UIParam.NSLICES),),
    ),
    Case(
        "bssfp_3d",
        "256 x 256 x 128",
        dict([_p(UIParam.NX, 256), _p(UIParam.NY, 256), _p(UIParam.NSLICES, 128)]),
        scaled=(str(UIParam.NY), str(UIParam.NSLICES)),
    ),
    Case(
        "zte_3d",
        "128, 16384 spokes",
        dict([_p(UIParam.NX, 128), _p(UIParam.NUM_SHOTS, 16384)]),
        scaled=(str(UIParam.NUM_SHOTS),),
    ),
    Case(
        "gre_noncart_3d",
        "256, 8192 interleaves",
        dict([_p(UIParam.NX, 256), _p(UIParam.NUM_SHOTS, 8192)]),
        scaled=(str(UIParam.NUM_SHOTS),),
    ),
]


# --------------------------------------------------------------------------
# Module-driven cases: the same chronologies, past the protocol caps.
# --------------------------------------------------------------------------


def _dim(value: int, scale: float) -> int:
    return max(1, round(value * scale))


def _gre_3d(scale):
    """Excitation + line readout, one shot per (ky, kz), spoiled."""
    system = pp.Opts(**SYSTEM_KW)
    fov, nx = (0.24, 0.24, 0.24), 512
    ny, nz = _dim(512, scale), _dim(512, scale)
    pulse = design.make_slice_selective_pulse(np.deg2rad(12), 0.16, system=system)
    line = design.make_line_readout(system, fov, (nx, ny, nz), slice_rephasing=pulse.rephasers)
    pulse = pulse.without_rephasers()
    lin = np.repeat(np.arange(ny), nz)
    par = np.tile(np.arange(nz), ny)
    phases = np.asarray(design.make_rf_spoiling_schedule(ny * nz, increment=np.deg2rad(117.0)))
    # The plugin's chronology is pulse, TE delay, readout, TR delay; the two
    # delays carry no library entry but are blocks the cursor still walks.
    te_delay, tr_delay = pp.make_delay(2e-3), pp.make_delay(3e-3)

    def build():
        seq = pp.Sequence(system)
        # One call per phase-encode step covers its partition loop, which is
        # how the plugin drives it.
        for start in range(0, ny * nz, nz):
            stop = start + nz
            seq.add_range(
                (pulse, {"phase_offset_rad": phases[start:stop]}),
                te_delay,
                (line, {"lin_idx": lin[start:stop], "par_idx": par[start:stop],
                        "phase_offset_rad": phases[start:stop]}),
                tr_delay,
            )
        return seq

    return f"{nx} x {ny} x {nz}", build


def _mprage_3d(scale):
    """Inversion, then one 1024-view GRE train per shot -- the largest case here."""
    system = pp.Opts(**SYSTEM_KW)
    fov, nx = (0.24, 0.24, 0.24), 512
    ny, nz = _dim(1024, scale), _dim(512, scale)
    etl = min(1024, ny)
    inversion = design.make_inversion_pulse(adiabatic=True, system=system)
    pulse = design.make_slice_selective_pulse(np.deg2rad(8), 0.16, system=system)
    line = design.make_line_readout(
        system, fov, (nx, ny, nz), slice_rephasing=pulse.rephasers, spoil_position="post"
    )
    pulse = pulse.without_rephasers()
    lin = np.repeat(np.arange(ny), nz)
    par = np.tile(np.arange(nz), ny)
    shots = (ny * nz) // etl
    phases = np.asarray(design.make_rf_spoiling_schedule(ny * nz, increment=np.deg2rad(117.0)))
    ti_delay = pp.make_delay(0.9)
    recovery = pp.make_delay(1.2)
    te_delay, tr_delay = pp.make_delay(2e-3), pp.make_delay(3e-3)

    def build():
        seq = pp.Sequence(system)
        for shot in range(shots):
            first, last = shot * etl, (shot + 1) * etl
            for block in inversion:
                seq.add_block(*block)
            seq.add_block(ti_delay)
            seq.add_range(
                (pulse, {"phase_offset_rad": phases[first:last]}),
                te_delay,
                (line, {"lin_idx": lin[first:last], "par_idx": par[first:last],
                        "phase_offset_rad": phases[first:last]}),
                tr_delay,
            )
            seq.add_block(recovery)
        return seq

    return f"{nx} x {ny} x {nz}, ETL {etl} ({shots} shots)", build


def _fse_3d(etl):
    """One refocusing train per shot; ``etl`` echoes each."""

    def make(scale):
        system = pp.Opts(**SYSTEM_KW)
        fov, nx = (0.24, 0.24, 0.24), 512
        ny, nz = _dim(512, scale), _dim(512, scale)
        train = min(etl, ny)
        refocusing = design.make_refocusing_pulse(slice_thickness=0.16, system=system)
        excite = design.make_slice_selective_pulse(np.deg2rad(90), 0.16, system=system)
        fse = design.make_fse_readout(system, fov, (nx, ny, nz), train, refocusing)

        per_partition = ny // train
        shots = per_partition * nz
        lin = np.stack([np.arange(train) * per_partition + (i % per_partition) for i in range(shots)])
        par = np.repeat(np.arange(nz) % nz, per_partition)[:, None] * np.ones(train, dtype=int)
        fse.set_state(lin_idx=lin[0], par_idx=par[0])

        def build():
            seq = pp.Sequence(system)
            seq.add_range(excite, (fse, {"lin_idx": lin, "par_idx": par}))
            return seq

        return f"{nx} x {ny} x {nz}, ETL {train} ({shots} shots)", build

    return make


def _epi_3d(scale):
    """One long blipped train per shot."""
    system = pp.Opts(**SYSTEM_KW)
    fov, nx = (0.24, 0.24, 0.24), 256
    ny, nz = _dim(256, scale), _dim(256, scale)
    excite = design.make_slice_selective_pulse(np.deg2rad(90), 0.16, system=system)
    mask = np.column_stack((np.arange(ny, dtype=int), np.zeros(ny, dtype=int)))
    epi = design.make_epi_readout(system, fov, (nx, ny, nz), 1, mask)
    shots = nz
    epi.set_state(lin_idx=0, par_idx=0)

    def build():
        seq = pp.Sequence(system)
        seq.add_range(
            excite,
            (epi, {"lin_idx": np.zeros(shots, dtype=int), "par_idx": np.arange(nz)}),
        )
        return seq

    return f"{nx} x {ny} x {nz}, ETL {epi.etl} ({shots} shots)", build


def _bssfp_3d(scale):
    """Balanced steady state: excitation and readout are one module."""
    system = pp.Opts(**SYSTEM_KW)
    fov, nx = (0.24, 0.24, 0.24), 256
    ny, nz = _dim(256, scale), _dim(256, scale)
    excitation = design.make_hard_pulse(np.deg2rad(35), duration=0.5e-3, system=system)
    # One segment per partition: each is a whole ky train, which is the shape
    # the plugin drives and the one worth registering in bulk.
    readout = design.make_bssfp_readout(system, fov, (nx, ny, nz), excitation, segment=nz)
    segments = np.arange(readout.num_segments)
    readout.set_state(segment_idx=0)

    def build():
        seq = pp.Sequence(system)
        seq.add_range((readout, {"segment_idx": segments}))
        return seq

    return f"{nx} x {ny} x {nz} ({readout.num_segments} segments)", build


def _gre_noncart_3d(scale):
    """Stack-of-spirals: one interleaf per shot, every partition."""
    system = pp.Opts(**SYSTEM_KW)
    fov, nx = (0.24, 0.24, 0.24), 256
    interleaves, nz = _dim(4096, scale), _dim(64, scale)
    pulse = design.make_slice_selective_pulse(np.deg2rad(12), 0.16, system=system)
    readout = design.make_spiral_stack_readout(
        system, fov, nx, interleaves, 0.24, nz, slice_rephasing=pulse.rephasers
    )
    pulse = pulse.without_rephasers()
    shots = interleaves * nz
    lin = np.tile(np.arange(interleaves), nz)
    par = np.repeat(np.arange(nz), interleaves)
    phases = np.asarray(design.make_rf_spoiling_schedule(shots, increment=np.deg2rad(117.0)))

    def build():
        seq = pp.Sequence(system)
        for start in range(0, shots, interleaves):
            stop = start + interleaves
            seq.add_range(
                (pulse, {"phase_offset_rad": phases[start:stop]}),
                (readout, {"lin_idx": lin[start:stop], "par_idx": par[start:stop],
                           "phase_offset_rad": phases[start:stop]}),
            )
        return seq

    return f"{nx}, {interleaves} interleaves x {nz} partitions", build


def _zte_3d(scale):
    """Rotation-encoded ZTE: one designed segment replayed under a rotation per shot."""
    system = pp.Opts(**SYSTEM_KW)
    nx, shots, views = 256, _dim(16384, scale), 16
    base, rotations = design.make_rotated_projection_sampling(
        (nx, nx, nx), shots=shots, views=views, scheme="spiral", require_fibonacci=False
    )
    excitation = design.make_hard_pulse(np.deg2rad(3), duration=8e-6, system=system, use="excitation")
    readout = design.make_zte_readout(
        system, 0.24, nx, base.flatten(), excitation, tr_s=8e-3, rotation_encoded=True
    )
    view_idx = np.arange(readout.num_views)
    lin = np.stack([shot * readout.num_views + view_idx for shot in range(shots)])
    readout.set_state(lin_idx=lin[0], rotation=rotations[0])

    def build():
        seq = pp.Sequence(system)
        seq.add_range((readout, {"lin_idx": lin, "rotation": rotations}))
        return seq

    return f"{nx}, {shots} shots x {readout.num_views} views", build


MODULE_CASES = [
    ModuleCase("mprage_3d", "512 x 1024 x 512, ETL 1024", _mprage_3d),
    ModuleCase("gre_3d", "512 x 512 x 512", _gre_3d),
    ModuleCase("fse_3d", "512³, ETL 128", _fse_3d(128)),
    ModuleCase("fse_3d", "512³, ETL 256", _fse_3d(256)),
    ModuleCase("epi_3d", "256³, one train per shot", _epi_3d),
    ModuleCase("bssfp_3d", "256 x 256 x 256", _bssfp_3d),
    ModuleCase("gre_noncart_3d", "spiral, 4096 x 64", _gre_noncart_3d),
    ModuleCase("zte_3d", "256, 16384 x 16 spokes", _zte_3d),
]


class _Sink:
    """Swallow the plugin's ``pio.write`` so only the design pass is timed."""

    def __enter__(self):
        self._real = pio.write
        pio.write = self
        return self

    def __exit__(self, *exc):
        pio.write = self._real
        return False

    def __call__(self, seq, output=None, **kwargs):  # noqa: ARG002 - pio.write's signature
        # The plugin hands the sequence over here and then drops it, so this is
        # the only moment the libraries can be counted before deduplication --
        # which is the count worth having, since it is what the design pass
        # actually held.
        self.blocks = len(seq.block_durations)
        self.events, self.extensions = _library_entries(seq)
        return None


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _library_entries(seq) -> tuple[int, int]:
    """``(block-referenced, extension-referenced)`` rows after the design pass."""
    counts = []
    for group in (_EVENT_LIBRARIES, _EXTENSION_LIBRARIES):
        total = 0
        for name in group:
            library = getattr(seq, name, None)
            if library is None:
                continue
            # The append-only libraries answer len(); shape_library and
            # extensions_library are upstream EventLibrary and only have .data.
            try:
                total += len(library)
            except TypeError:
                total += len(library.data)
        counts.append(total)
    return counts[0], counts[1]


def run_module_case(case: ModuleCase, scale: float, repeats: int) -> dict:
    """Time one module-driven chronology, hoisting the module design out."""
    try:
        size, build = case.build(scale)
    except Exception as exc:
        return {"plugin": case.plugin, "label": case.label, "error": f"{type(exc).__name__}: {exc}"}
    label = size if scale == 1.0 else f"{size} @{scale:g}"

    best, blocks, events, extensions = None, 0, 0, 0
    for _ in range(repeats):
        gc.collect()
        before = _peak_rss_mb()
        try:
            start = time.perf_counter()
            seq = build()
            elapsed = time.perf_counter() - start
        except Exception as exc:
            return {"plugin": case.plugin, "label": label, "error": f"{type(exc).__name__}: {exc}"}
        blocks = len(seq.block_durations)
        events, extensions = _library_entries(seq)
        row = {"design_s": elapsed, "rss_delta_mb": _peak_rss_mb() - before}
        if best is None or row["design_s"] < best["design_s"]:
            best = row
        del seq
        gc.collect()

    return {
        "plugin": case.plugin,
        "family": case.family,
        "label": label,
        "scale": scale,
        "blocks": blocks,
        "events": events,
        "extensions": extensions,
        "us_per_block": best["design_s"] / blocks * 1e6 if blocks else 0.0,
        **best,
    }


def run_case(case: Case, scale: float, repeats: int) -> dict:
    plugin = load_plugin(case.plugin)
    opts = pp.Opts()
    protocol = plugin.get_default_protocol(opts)
    overrides = dict(case.overrides)
    label = case.label
    if scale != 1.0:
        for key in case.scaled:
            overrides[key] = max(1, round(overrides[key] * scale))
        label = f"{case.label} @{scale:g} -> " + ", ".join(f"{k}={overrides[k]}" for k in case.scaled)
    for key, value in overrides.items():
        if key not in protocol:
            return {"plugin": case.plugin, "label": label, "error": f"no protocol entry {key!r}"}
        protocol[key]["value"] = value

    try:
        verdict = plugin.validate_protocol(opts, protocol)
    except Exception as exc:
        return {"plugin": case.plugin, "label": label, "error": f"{type(exc).__name__}: {exc}"}
    if not verdict.get("valid", False):
        return {"plugin": case.plugin, "label": label, "error": verdict.get("info", "invalid protocol")}

    best = None
    blocks = events = extensions = 0
    for _ in range(repeats):
        gc.collect()
        before = _peak_rss_mb()
        try:
            with _Sink() as sink:
                start = time.perf_counter()
                plugin.make_sequence(opts, protocol, None)
                elapsed = time.perf_counter() - start
        except Exception as exc:
            # A scaled-down sweep can land on a protocol whose gradients do not
            # solve — an infeasible corner is a property of the protocol, not
            # something the sweep should stop for.
            return {"plugin": case.plugin, "label": label, "error": f"{type(exc).__name__}: {exc}"}
        blocks, events, extensions = sink.blocks, sink.events, sink.extensions
        row = {"design_s": elapsed, "rss_delta_mb": _peak_rss_mb() - before}
        if best is None or row["design_s"] < best["design_s"]:
            best = row
        gc.collect()

    return {
        "plugin": case.plugin,
        "family": case.family,
        "label": label,
        "scale": scale,
        "blocks": blocks,
        "events": events,
        "extensions": extensions,
        "duration_s": verdict.get("duration"),
        "us_per_block": best["design_s"] / blocks * 1e6 if blocks else 0.0,
        **best,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run one plugin only")
    parser.add_argument("--family", default="all", choices=("all", "protocol", "module"))
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the scaled dimensions")
    parser.add_argument("--repeats", type=int, default=1, help="timings are the minimum of this many")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    selected = []
    if args.family in ("all", "protocol"):
        selected += [(case, run_case) for case in CASES]
    if args.family in ("all", "module"):
        selected += [(case, run_module_case) for case in MODULE_CASES]

    results = []
    header = (
        f"{'plugin':<16} {'size':<34} {'blocks':>10} {'design':>9} "
        f"{'us/blk':>8} {'RSS MB':>8} {'events':>9} {'ext rows':>10}"
    )
    current = None
    for case, runner in selected:
        if args.only and case.plugin != args.only:
            continue
        if case.family != current:
            current = case.family
            title = "protocol maximum" if current == "protocol" else "modules driven past the protocol caps"
            print(f"\n{title}\n{header}\n{'-' * len(header)}")
        row = runner(case, args.scale, args.repeats)
        results.append(row)
        if "error" in row:
            print(f"{row['plugin']:<16} {row['label']:<34} {row['error']}", flush=True)
        else:
            print(
                f"{row['plugin']:<16} {row['label']:<34} {row['blocks']:>10,} "
                f"{row['design_s']:8.2f}s {row['us_per_block']:8.2f} {row['rss_delta_mb']:8.0f} "
                f"{row.get('events', 0):>9,} {row.get('extensions', 0):>10,}",
                flush=True,
            )

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
