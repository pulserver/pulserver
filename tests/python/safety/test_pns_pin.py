"""The gate's number itself, pinned.

Every other PNS test asserts two implementations agree; all of them stay
green when the model, the extraction and the assembly move *together*. These
tests hold the worst-case Irnich peak of every corpus sequence against the
value recorded in ``tests/utils/expected/pns_peaks.json``, so a verdict can
only change through a regeneration diff someone reviews.
"""

import json

import pns_pin
import pytest

from .conftest import CORPUS, EXPECTED

_RECORD_PATH = EXPECTED / pns_pin.EXPECTED_NAME
_RECORD = json.loads(_RECORD_PATH.read_text())


def test_every_corpus_sequence_is_pinned():
    on_disk = {p.stem for p in CORPUS.glob("*.seq")}
    recorded = set(_RECORD["peaks"])
    assert on_disk == recorded, (
        f"corpus and recording disagree (run tests/utils/generate_fixtures.py): "
        f"unpinned={sorted(on_disk - recorded)} stale={sorted(recorded - on_disk)}"
    )


def test_the_recorded_system_is_the_pinning_system():
    assert _RECORD["irnich"] == pns_pin.IRNICH
    assert _RECORD["system"] == pns_pin._system_record(pns_pin.pin_system())


@pytest.mark.parametrize("name", sorted(_RECORD["peaks"]))
def test_the_gate_peak_matches_its_recorded_value(name):
    recorded = _RECORD["peaks"][name]
    computed = pns_pin.compute_peak(CORPUS / f"{name}.seq")
    if recorded is None:
        assert computed is None
    elif recorded == 0.0:
        assert computed == 0.0
    else:
        assert computed == pytest.approx(recorded, rel=pns_pin.RTOL), (
            f"{name}: gate peak {computed!r} moved from recorded {recorded!r} "
            f"({(computed - recorded) / recorded * 100.0:+.3f}%)"
        )
