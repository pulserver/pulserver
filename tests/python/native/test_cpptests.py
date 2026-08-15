"""The C++ (GoogleTest) cases, one pytest node per case.

Assertions and fixtures live in ``tests/cpptests``; this lane discovers each
binary's cases with ``--gtest_list_tests`` and runs them one at a time with
``--gtest_filter``, the same per-case processes CTest launches. Build policy
and discovery are in ``_lanes.py``.
"""

from __future__ import annotations

import subprocess

import pytest

from _lanes import CPPTESTS_DIR, ROOT, gtest_cases

REASON, CASES = gtest_cases()


@pytest.mark.parametrize(
    ("binary", "case"),
    CASES or [("lane", "unavailable")],
    ids=[f"{binary}:{case}" for binary, case in CASES] or ["lane-unavailable"],
)
def test_the_cpp_case_passes(binary, case):
    if REASON:
        pytest.skip(REASON)
    result = subprocess.run(
        [str(CPPTESTS_DIR / binary), f"--gtest_filter={case}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{binary} {case} failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}{result.stderr[-2000:]}"
    )
