"""The calibration a stream computes once and reuses: :meth:`ReconContext.memoized`."""

from __future__ import annotations

import logging

from pulserver import recon


def test_a_value_is_computed_once_for_its_key_and_reused_after():
    context = recon.ReconContext.offline()
    calls = []

    def compute():
        calls.append(None)
        return len(calls)

    first = context.memoized(("maps", 0), compute)
    again = context.memoized(("maps", 0), compute)
    other = context.memoized(("maps", 1), compute)

    assert (first, again, other) == (1, 1, 2)
    assert len(calls) == 2


def test_the_values_of_one_stream_are_not_those_of_another():
    first, second = recon.ReconContext.offline(), recon.ReconContext.offline()

    first.memoized("kernel", lambda: "first")

    assert second.memoized("kernel", lambda: "second") == "second"


def test_a_named_value_is_logged_with_its_time_once_when_computed(caplog):
    context = recon.ReconContext.offline()

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        context.memoized("field", lambda: 1, what="distortion field")
        context.memoized("field", lambda: 2, what="distortion field")

    records = [r.getMessage() for r in caplog.records if "distortion" in r.getMessage()]
    assert len(records) == 1
    assert records[0].startswith("distortion field computed in ")
