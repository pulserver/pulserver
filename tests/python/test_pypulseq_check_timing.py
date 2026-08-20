"""``check_timing`` against upstream PyPulseq's own checker.

The check is compiled and reads the libraries rather than decoding blocks, so
what has to be shown is that it answers exactly what upstream answers -- the
same findings, in the same order, carrying the same fields.

Deliberately broken files are the point here: a corpus that passes only shows
the check invents nothing. Each case below violates one rule, written as exact
``.seq`` text because no builder should be able to express it.
"""

from __future__ import annotations

from pathlib import Path

import pulserver.pypulseq as pp
import pytest

from pulserver.pypulseq._pulseqpp import to_upstream

from fixture_corpus import FIXTURES_DIR

_HEADER = """# Pulseq sequence file
# Written by tests/python/test_pypulseq_check_timing.py

[VERSION]
major 1
minor 5
revision 1

[DEFINITIONS]
AdcRasterTime 1e-07
BlockDurationRaster 1e-05
GradientRasterTime 1e-05
RadiofrequencyRasterTime 1e-06

"""

#: Flat magnitude, zero phase, and a time ramp giving the pulse its duration.
_RF_SHAPES = """[SHAPES]
shape_id 1
num_samples 2
1
1
shape_id 2
num_samples 2
0
0
shape_id 3
num_samples 2
0
1000
"""

#: name -> (body, system overrides). Each violates one rule.
BROKEN = {
    "a trapezoid ramp off the raster": (
        """[BLOCKS]
1 100   0   1   0   0  0  0

[TRAP]
 1          0.2  15  970  15   0
""",
        {},
    ),
    "a negative gradient delay": (
        """[BLOCKS]
1 100   0   1   0   0  0  0

[TRAP]
 1          0.2  20  960  20   -10
""",
        {},
    ),
    "a block shorter than its gradient": (
        """[BLOCKS]
1  50   0   1   0   0  0  0

[TRAP]
 1          0.2  20  960  20   0
""",
        {},
    ),
    "an rf delay off the raster": (
        """[BLOCKS]
1 100   1   0   0   0  0  0

[RF]
1          500 1 2 3 500 0.5 0 0 0 0 e

"""
        + _RF_SHAPES,
        {},
    ),
    "an rf pulse running past its block": (
        """[BLOCKS]
1  50   1   0   0   0  0  0

[RF]
1          500 1 2 3 500 0 0 0 0 0 e

"""
        + _RF_SHAPES,
        {},
    ),
    "an adc dwell off the raster": (
        """[BLOCKS]
1 100   0   0   0   0  1  0

[ADC]
1 100 350 0 0 0 0 0 0
""",
        {},
    ),
    "an adc running past its block": (
        """[BLOCKS]
1  20   0   0   0   0  1  0

[ADC]
1 1000 1000 0 0 0 0 0 0
""",
        {},
    ),
    "an rf pulse inside the coil's dead time": (
        """[BLOCKS]
1 100   1   0   0   0  0  0

[RF]
1          500 1 2 3 500 0 0 0 0 0 e

"""
        + _RF_SHAPES,
        {"rf_dead_time": 100e-6, "rf_ringdown_time": 30e-6},
    ),
    "an adc inside the receive dead time": (
        """[BLOCKS]
1 100   0   0   0   0  1  0

[ADC]
1 100 350 0 0 0 0 0 0
""",
        {"adc_dead_time": 20e-6},
    ),
}


def _read(tmp_path: Path, name: str, body: str, system: dict) -> pp.Sequence:
    """The file at ``body``, read into a sequence with its own ``Opts``.

    Its own, because a default-constructed sequence shares one, and a case
    that sets a dead time would otherwise set it for every later case too.
    """
    path = tmp_path / f"{abs(hash(name))}.seq"
    path.write_text(_HEADER + body)
    seq = pp.Sequence(system=pp.Opts(**system))
    seq.read(str(path))
    return seq


def _reports(seq: pp.Sequence):
    """This checker's report and upstream's, both as plain dictionaries."""
    mine = [vars(entry) for entry in seq.check_timing()[1]]
    theirs = [vars(entry) for entry in to_upstream(seq).check_timing()[1]]
    return mine, theirs


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_a_broken_sequence_gets_upstreams_report_exactly(case, tmp_path):
    body, system = BROKEN[case]
    mine, theirs = _reports(_read(tmp_path, case, body, system))

    assert theirs, (
        "the case must actually break something for the comparison to mean anything"
    )
    assert mine == theirs


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES_DIR.glob("*.seq")))
def test_a_zoo_sequence_is_clean_and_agrees_with_upstream(name):
    seq = pp.Sequence(system=pp.Opts())
    seq.read(str(FIXTURES_DIR / name))
    mine, theirs = _reports(seq)

    assert mine == theirs
    assert not mine


def test_a_window_is_judged_the_way_upstream_judges_that_window(tmp_path):
    """``time_range`` restricts which blocks answer, not what they are asked."""
    body = """[BLOCKS]
1 100   0   1   0   0  0  0
2 100   0   2   0   0  0  0

[TRAP]
 1          0.2  20  960  20   0
 2          0.2  15  970  15   0
"""
    seq = _read(tmp_path, "window", body, {})
    whole = [vars(e) for e in seq.check_timing()[1]]
    assert {entry["block"] for entry in whole} == {2}

    first_only = seq.check_timing(time_range=[0.0, 1e-3])[1]
    assert first_only == []


def test_the_verdict_and_the_report_agree(tmp_path):
    body, system = BROKEN["a trapezoid ramp off the raster"]
    seq = _read(tmp_path, "verdict", body, system)

    is_ok, report = seq.check_timing()

    assert not is_ok
    assert report

    clean = pp.Sequence(system=pp.Opts())
    clean.read(str(FIXTURES_DIR / "gre_2d.seq"))
    assert clean.check_timing() == (True, [])
