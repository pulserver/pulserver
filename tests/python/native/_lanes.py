"""Discovery and build policy for the native (C and C++) test lanes.

The lanes reuse the compiled suites as-is: the minunit runner
(``tests/ctests/build/bin/run_tests``) exposes its suites via ``--list`` and
runs one per argument; each GoogleTest binary under ``tests/cpptests/build``
enumerates its cases via ``--gtest_list_tests`` and runs one via
``--gtest_filter``. pytest contributes discovery, per-case nodes, and
reporting on top.

Build policy: an existing binary is used as found -- rebuilding after a C
change stays an explicit ``scripts/build_ctests.sh`` /
``scripts/build_cpp_tests.sh`` step. A missing binary skips its lane with
instructions, unless ``PULSERVER_BUILD_NATIVE=1`` asks collection to run the
build script itself. ``PULSERVER_SKIP_NATIVE=1`` skips both lanes outright.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CTESTS_BINARY = ROOT / "tests" / "ctests" / "build" / "bin" / "run_tests"
CPPTESTS_DIR = ROOT / "tests" / "cpptests" / "build"


def _ensure_built(binary: Path, script: str) -> str | None:
    """Return a skip reason, or None once ``binary`` is available."""
    if os.environ.get("PULSERVER_SKIP_NATIVE") == "1":
        return "PULSERVER_SKIP_NATIVE=1"
    if binary.exists():
        return None
    if os.environ.get("PULSERVER_BUILD_NATIVE") != "1":
        return (
            f"{binary.relative_to(ROOT)} not built -- run `bash {script}`,"
            " or set PULSERVER_BUILD_NATIVE=1 to build during collection"
        )
    if shutil.which("cmake") is None:
        return "cmake not available to build the native tests"
    build = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True
    )
    if build.returncode != 0:
        return f"`bash {script}` failed:\n{build.stdout[-2000:]}{build.stderr[-2000:]}"
    if not binary.exists():
        return f"`bash {script}` succeeded but {binary.relative_to(ROOT)} is still missing"
    return None


def ctest_suites() -> tuple[str | None, list[str]]:
    """The minunit suite names, or a reason the lane cannot run."""
    reason = _ensure_built(CTESTS_BINARY, "scripts/build_ctests.sh")
    if reason:
        return reason, []
    listing = subprocess.run(
        [str(CTESTS_BINARY), "--list"], capture_output=True, text=True
    )
    if listing.returncode != 0:
        return f"{CTESTS_BINARY.name} --list failed:\n{listing.stderr}", []
    return None, listing.stdout.split()


def _gtest_binaries() -> list[Path]:
    return sorted(
        path
        for path in CPPTESTS_DIR.glob("test_*")
        if path.is_file() and os.access(path, os.X_OK)
    )


def _gtest_cases(binary: Path) -> list[str]:
    listing = subprocess.run(
        [str(binary), "--gtest_list_tests"], capture_output=True, text=True
    )
    cases: list[str] = []
    prefix = ""
    for line in listing.stdout.splitlines():
        if not line or line.startswith("Running main"):
            continue
        name = line.split("#", 1)[0].rstrip()
        if not name:
            continue
        if not name.startswith(" "):
            prefix = name.strip()
        else:
            cases.append(prefix + name.strip())
    return cases


def gtest_cases() -> tuple[str | None, list[tuple[str, str]]]:
    """Every (binary name, case name) pair, or a reason the lane cannot run."""
    probe = CPPTESTS_DIR / "test_pulseq_file"
    reason = _ensure_built(probe, "scripts/build_cpp_tests.sh")
    if reason:
        return reason, []
    return None, [
        (binary.name, case)
        for binary in _gtest_binaries()
        for case in _gtest_cases(binary)
    ]
