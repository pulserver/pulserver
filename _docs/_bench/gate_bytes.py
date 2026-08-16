"""Byte-identity gate: every zoo plugin's ``.seq`` payload, hashed.

Documentation-only tooling, like everything else in this directory — but it is
the gate the sequence-writer work is held to, not a measurement. Event IDs are
handed out in visit order and ``remove_duplicates`` picks representatives by
first occurrence, so a change to *how* blocks are registered can leave the
sequence physically identical while changing the file — which stales every
``.pge`` truth fixture. Nothing about registration should change the bytes, and
this is what proves it.

Run it as::

    python docs/_bench/gate_bytes.py check      # diff against the baseline
    python docs/_bench/gate_bytes.py save       # re-baseline (deliberately)

``save`` rewrites ``gate_bytes_baseline.json``. Re-baselining is a decision, not
a fix: do it only when a change to the emitted file is intended and understood,
and say so in the commit.

Each case is one plugin at one protocol. Every plugin appears at its default
protocol, and the families with a batched or looped shot structure appear again
at a size past ``_MIN_BULK_SHOTS`` so the fast paths are actually exercised.

Timings are printed but not asserted — they are there to make a large
regression visible while the gate is being run for its real purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOO = REPO_ROOT / "examples" / "sequences"
BASELINE = Path(__file__).resolve().parent / "gate_bytes_baseline.json"

import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import UIParam

P = str


#: Default protocol for every plugin in the zoo, plus one larger size for each
#: family whose registration path depends on the shot count.
def _cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = [
        (path.stem, {}) for path in sorted(ZOO.glob("*.py")) if not path.stem.startswith("_")
    ]
    cases += [
        ("gre_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NSLICES): 4}),
        ("gre_3d", {P(UIParam.NX): 64, P(UIParam.NY): 32, P(UIParam.NSLICES): 16, P(UIParam.TR): 150.0}),
        ("mprage_3d", {P(UIParam.NX): 64, P(UIParam.NY): 32, P(UIParam.NSLICES): 16}),
        ("fse_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NSLICES): 4, P(UIParam.ETL): 8}),
        ("fse_3d", {P(UIParam.NX): 64, P(UIParam.NY): 32, P(UIParam.NSLICES): 16, P(UIParam.ETL): 8}),
        ("epi_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NSLICES): 8, P(UIParam.ETL): 8}),
        ("epi_3d", {P(UIParam.NX): 64, P(UIParam.NY): 32, P(UIParam.NSLICES): 16, P(UIParam.ETL): 8}),
        ("bssfp_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NUM_FRAMES): 4}),
        ("bssfp_3d", {P(UIParam.NX): 64, P(UIParam.NY): 32, P(UIParam.NSLICES): 8}),
        ("zte_3d", {P(UIParam.NX): 64, P(UIParam.NUM_SHOTS): 64}),
        ("gre_noncart_2d", {P(UIParam.NX): 64, P(UIParam.NUM_SHOTS): 16, P(UIParam.NSLICES): 4, P(UIParam.TR): 40.0}),
        ("gre_noncart_3d", {P(UIParam.NX): 64, P(UIParam.NUM_SHOTS): 16}),
        ("gre_radial_2d", {P(UIParam.NX): 64, P(UIParam.NUM_SHOTS): 32, P(UIParam.NSLICES): 4, P(UIParam.TR): 30.0}),
        ("gre_multiecho_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NSLICES): 4}),
        ("mprage_2d", {P(UIParam.NX): 64, P(UIParam.NY): 64, P(UIParam.NSLICES): 4}),
    ]
    return cases


_LOADED: dict[str, object] = {}


def load_plugin(name: str):
    """Exec-load a zoo plugin by file, the way the bridge does."""
    if name not in _LOADED:
        spec = importlib.util.spec_from_file_location(f"_gate_{name}", ZOO / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _LOADED[name] = module.PLUGIN
    return _LOADED[name]


class _Capture:
    """Take the payload the plugin hands to ``pio.write`` without touching disk."""

    def __enter__(self):
        self._real = pio.write
        pio.write = self
        self.payload = b""
        return self

    def __exit__(self, *exc):
        pio.write = self._real
        return False

    def __call__(self, seq, output=None, **kwargs):  # noqa: ARG002 - pio.write's signature
        self.payload = self._real(seq, output=None, **kwargs)
        return None


def run() -> dict:
    results: dict = {}
    for name, overrides in _cases():
        key = f"{name}|{json.dumps(overrides, sort_keys=True)}"
        try:
            plugin = load_plugin(name)
            opts = pp.Opts()
            protocol = plugin.get_default_protocol(opts)
            missing = [k for k in overrides if k not in protocol]
            if missing:
                results[key] = f"NOKEY:{missing[0]}"
                continue
            for param, value in overrides.items():
                protocol[param]["value"] = value

            verdict = plugin.validate_protocol(opts, protocol)
            if not verdict.get("valid", False):
                results[key] = f"INVALID:{verdict.get('info', '')}"
                continue

            start = time.perf_counter()
            with _Capture() as capture:
                plugin.make_sequence(opts, protocol, "/dev/null")
            results[key] = {
                "sha": hashlib.sha256(capture.payload).hexdigest()[:16],
                "bytes": len(capture.payload),
                "s": round(time.perf_counter() - start, 3),
            }
        except Exception as exc:  # a broken plugin is a gate result, not a crash
            results[key] = f"ERROR:{type(exc).__name__}:{exc}"
    return results


def _sha(entry) -> str:
    return entry["sha"] if isinstance(entry, dict) else str(entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "save"), nargs="?", default="check")
    args = parser.parse_args()

    print(f"pulserver: {Path(pio.__file__).resolve()}\n")
    results = run()

    if args.mode == "save":
        BASELINE.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
        for key, value in sorted(results.items()):
            print(f"{key:70s} {value}")
        print(f"\nsaved {len(results)} cases to {BASELINE.name}")
        return 0

    if not BASELINE.exists():
        print(f"no baseline at {BASELINE}; run `gate_bytes.py save` first")
        return 2

    baseline = json.loads(BASELINE.read_text())
    differing = 0
    for key in sorted(set(baseline) | set(results)):
        before, after = baseline.get(key), results.get(key)
        if _sha(before) != _sha(after):
            differing += 1
            print(f"DIFF {key}\n  baseline={before}\n  now     ={after}")
        else:
            was = before["s"] if isinstance(before, dict) else 0
            now = after["s"] if isinstance(after, dict) else 0
            speed = f"  {was:6.3f} -> {now:6.3f}s  ({now / was:.2f}x)" if was else ""
            print(f"ok   {key:66s}{speed}")

    if differing:
        print(f"\n*** {differing} of {len(baseline)} DIFFER ***")
        return 1
    print(f"\n{len(baseline)}/{len(baseline)} identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
