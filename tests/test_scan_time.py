"""The scan time and the segment instances of a collection, counted by the C library."""

import subprocess

from conftest import _build


def test_the_library_counts_the_segments_a_scan_plays_and_sums_its_time_exactly(
    tmp_path,
):
    """Hand-written collections: the cases are spelled out in the program."""
    program = _build(tmp_path, "check_segment_plays.c", "check_segment_plays")
    result = subprocess.run([str(program)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "ALL PASS" in result.stdout
