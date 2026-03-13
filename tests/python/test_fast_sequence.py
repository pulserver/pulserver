from __future__ import annotations

from io import BytesIO

import pypulseq as pp
import pytest
from pulserver import FastSequence, write


def _make_simple_seq(seq):
    g = pp.make_trapezoid(channel="x", amplitude=0.8 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    adc = pp.make_adc(num_samples=16, duration=1e-3, delay=0.0, system=seq.system)
    seq.add_block(g, adc)
    seq.add_block(pp.make_delay(1e-3))


def test_fast_sequence_disables_positional_set_block():
    seq = FastSequence()
    with pytest.raises(NotImplementedError):
        seq.set_block(1, pp.make_delay(1e-3))


def test_fast_sequence_add_block_without_dedup():
    seq = FastSequence(disable_event_dedup=True)
    trap1 = pp.make_trapezoid(channel="x", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    trap2 = pp.make_trapezoid(channel="x", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    seq.add_block(trap1)
    seq.add_block(trap2)

    # Same event inserted twice without dedup during build.
    assert len(seq.grad_library.data) == 2


def test_write_helper_returns_binary_payload():
    seq = FastSequence()
    _make_simple_seq(seq)

    payload = write(seq, output=None, check_timing=False)

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert b"[VERSION]" in payload


def test_write_helper_supports_binary_stream():
    seq = FastSequence()
    _make_simple_seq(seq)

    stream = BytesIO()
    signature = write(seq, output=stream, create_signature=True, check_timing=False)

    assert isinstance(signature, str)
    assert len(stream.getvalue()) > 0
