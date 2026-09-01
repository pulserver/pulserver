"""The fused arbitrary-gradient factory is upstream's, field for field.

``make_arbitrary_grad`` builds its event in one compiled pass. These tests
hold that path — and the Python reference path behind it — against upstream
PyPulseq: every field of the converted event equal, every refusal the same
exception with the same text.
"""

import numpy as np
import pypulseq as upstream
import pytest

import pulserver.pypulseq as pp
from pulserver.pypulseq._events import _make_arbitrary_grad, convert


def _system():
    return pp.Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
    )


def _wave(n, scale=0.08):
    t = np.linspace(0.0, 1.0, n)
    system = _system()
    w = np.sin(6 * np.pi * t) * 4.0 * t * (1.0 - t)
    return w * (scale * system.max_grad / np.abs(w).max())


def _assert_same_event(ours, theirs):
    assert type(ours).__name__ == type(theirs).__name__ == "GradEvent"
    assert ours.channel == theirs.channel
    assert ours.amplitude == theirs.amplitude
    assert ours.first == theirs.first
    assert ours.last == theirs.last
    assert ours.delay == theirs.delay
    assert np.array_equal(np.asarray(ours.waveform), np.asarray(theirs.waveform))
    assert np.array_equal(np.asarray(ours.tt), np.asarray(theirs.tt))


CASES = [
    {},
    {"channel": "y"},
    {"channel": "z", "delay": 40e-6},
    {"first": 1234.5},
    {"last": -987.0},
    {"first": 0.0, "last": 0.0},
    {"oversampling": True, "n": 257},
    {"oversampling": True, "n": 257, "first": 11.0, "last": -3.0},
]


@pytest.mark.parametrize("case", CASES, ids=[str(sorted(c)) for c in CASES])
def test_the_fast_path_builds_upstreams_event(case):
    case = dict(case)
    n = case.pop("n", 256)
    kwargs = {"channel": "x", "waveform": _wave(n), "system": _system(), **case}
    ours = pp.make_arbitrary_grad(**kwargs)
    theirs = convert(upstream.make_arbitrary_grad(**kwargs))
    _assert_same_event(ours, theirs)


def test_a_float32_waveform_builds_the_same_event():
    # Upstream extrapolates the edges in the waveform's own dtype; the fused
    # pass works in double. The samples convert identically, the scalar edges
    # agree to float32 resolution.
    kwargs = {
        "channel": "x",
        "waveform": _wave(256).astype(np.float32),
        "system": _system(),
    }
    ours = pp.make_arbitrary_grad(**kwargs)
    theirs = convert(upstream.make_arbitrary_grad(**kwargs))
    assert type(ours).__name__ == type(theirs).__name__ == "GradEvent"
    assert np.array_equal(np.asarray(ours.waveform), np.asarray(theirs.waveform))
    assert np.array_equal(np.asarray(ours.tt), np.asarray(theirs.tt))
    assert ours.amplitude == pytest.approx(theirs.amplitude, rel=1e-6)
    assert ours.first == pytest.approx(theirs.first, rel=1e-6)
    assert ours.last == pytest.approx(theirs.last, rel=1e-6)


def test_the_reference_path_builds_the_same_event():
    kwargs = {"channel": "x", "waveform": _wave(256), "system": _system()}
    fast = pp.make_arbitrary_grad(**kwargs)
    reference = convert(_make_arbitrary_grad(**kwargs))
    _assert_same_event(fast, reference)


def test_a_list_waveform_is_refused_on_both_sides():
    # Not a documented input; each implementation trips over it at its own
    # line, so the shared contract is the refusal, not its wording.
    w = list(_wave(256))
    with pytest.raises(TypeError):
        upstream.make_arbitrary_grad(channel="x", waveform=w, system=_system())
    with pytest.raises(TypeError):
        pp.make_arbitrary_grad(channel="x", waveform=w, system=_system())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel": "q", "waveform": np.zeros(8)},
        {"channel": "x", "waveform": np.full(64, 1e9)},
        {
            "channel": "x",
            "waveform": np.concatenate([np.zeros(32), np.full(32, 2.0e6)]),
        },
        {"channel": "x", "waveform": np.zeros(64), "oversampling": True},
    ],
    ids=["bad_channel", "amplitude", "slew", "even_oversampled"],
)
def test_every_refusal_is_upstreams_to_the_character(kwargs):
    kwargs = {"system": _system(), **kwargs}
    with pytest.raises(ValueError) as theirs:
        upstream.make_arbitrary_grad(**kwargs)
    with pytest.raises(ValueError) as ours:
        pp.make_arbitrary_grad(**kwargs)
    assert str(ours.value) == str(theirs.value)
