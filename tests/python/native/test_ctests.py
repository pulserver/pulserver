"""The C (minunit) suites, one pytest node per suite.

Assertions and fixtures live in ``tests/ctests``; this lane runs each suite
through the compiled runner and surfaces its output on failure. Build policy
and discovery are in ``_lanes.py``.
"""

from __future__ import annotations

import subprocess

import pytest

from _lanes import CTESTS_BINARY, ROOT, ctest_suites

REASON, SUITES = ctest_suites()


@pytest.mark.parametrize("suite", SUITES or ["lane-unavailable"])
def test_the_c_suite_passes(suite):
    if REASON:
        pytest.skip(REASON)
    result = subprocess.run(
        [str(CTESTS_BINARY), suite], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"{suite} failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}{result.stderr[-2000:]}"
    )
