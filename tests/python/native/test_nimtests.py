"""The Nim bridge tests, as one pytest node.

The bridge suite (``tests/nim/run_tests.sh``) builds the installer bundle,
runs the ``nimble test`` cases, and exercises the compiled host end to end.
That is one long process with its own environment bootstrap, so it stays a
single node rather than per-case ones.

Skips when no Nim >= 2 toolchain is on PATH, or when
``PULSERVER_SKIP_NIM=1``. The first run builds the test bundle (network
access for the installer); later runs reuse it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from _lanes import ROOT


def _skip_reason() -> str | None:
    if os.environ.get("PULSERVER_SKIP_NIM") == "1":
        return "PULSERVER_SKIP_NIM=1"
    if shutil.which("nim") is None or shutil.which("nimble") is None:
        return "nim/nimble not on PATH"
    return None


def test_the_nim_bridge_suite_passes():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    result = subprocess.run(
        ["bash", "tests/nim/run_tests.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"nim bridge suite failed (exit {result.returncode}):\n"
        f"{result.stdout[-6000:]}{result.stderr[-3000:]}"
    )
