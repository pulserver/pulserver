"""The whole PyPulseq namespace works on our events, not only ``make_*``.

Upstream's helpers do not merely read attributes off what they are handed:
``calc_duration`` and ``add_gradients`` run ``isinstance(x, SimpleNamespace)``
checks, and ``scale_grad``, ``rotate`` and ``split_gradient_at`` run
``copy.deepcopy``. A slotted C++ event fails both, so before the
interoperation decorator was applied across the namespace, seven of the eight
functions exercised here raised ``TypeError`` on our own events -- including
``calc_duration``, which almost every sequence script calls.
"""

from __future__ import annotations

import warnings

import numpy as np
import pypulseq as upstream
import pytest

import pulserver.pypulseq as pp
from pulserver._ext import _pulseqpp_wrapper as cxx


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def pair(system):
    """The same events built by both packages."""

    def build(make):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "gx": make.make_trapezoid(channel="x", area=1000, duration=2e-3, system=system),
                "gx2": make.make_trapezoid(channel="x", area=200, duration=1e-3, system=system),
                "gy": make.make_trapezoid(channel="y", area=500, duration=1e-3, system=system),
                "rf": make.make_sinc_pulse(flip_angle=0.5, duration=2e-3, system=system),
                "adc": make.make_adc(num_samples=64, duration=1.6e-3, system=system),
                "trig": make.make_trigger("physio1", duration=1e-3),
            }

    return build(pp), build(upstream)


def _events_in(value):
    if isinstance(value, (list, tuple)):
        return [e for item in value for e in _events_in(item)]
    return [value] if hasattr(value, "type") else []


# %% every namespace helper accepts our events and returns ours


def test_calc_duration_accepts_slotted_events(pair, system):
    ours, theirs = pair
    assert pp.calc_duration(ours["gx"]) == upstream.calc_duration(theirs["gx"])
    assert pp.calc_duration(ours["gx"], ours["adc"]) == upstream.calc_duration(
        theirs["gx"], theirs["adc"]
    )


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("scale_grad", lambda m, e, s: m.scale_grad(e["gx"], 0.5)),
        ("add_gradients", lambda m, e, s: m.add_gradients([e["gx"], e["gx2"]], system=s)),
        ("split_gradient", lambda m, e, s: m.split_gradient(e["gx"], system=s)),
        ("split_gradient_at", lambda m, e, s: m.split_gradient_at(e["gx"], 1e-3, system=s)),
        ("rotate", lambda m, e, s: m.rotate(e["gx"], e["gy"], angle=0.3, axis="z", system=s)),
        ("align", lambda m, e, s: m.align(right=[e["gx"]], left=[e["gy"]])),
    ],
)
def test_the_helpers_agree_with_upstream_and_hand_back_slotted_events(pair, system, name, call):
    ours, theirs = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = call(pp, ours, system)
        yours = call(upstream, theirs, system)

    produced = _events_in(mine)
    assert produced, f"{name} returned no events"
    assert all(isinstance(event, cxx.Event) for event in produced), (
        f"{name} handed back namespaces, not slotted events"
    )

    for got, want in zip(produced, _events_in(yours)):
        assert got.type == want.type
        for field in ("amplitude", "delay", "rise_time", "flat_time", "fall_time", "area"):
            if hasattr(want, field) and hasattr(got, field):
                np.testing.assert_allclose(
                    getattr(got, field), getattr(want, field), rtol=1e-12, err_msg=f"{name}.{field}"
                )
        if hasattr(want, "waveform") and hasattr(got, "waveform"):
            np.testing.assert_allclose(got.waveform, want.waveform, rtol=1e-10)


# %% the converter underneath


def test_as_namespace_completes_the_fields_upstream_reads(pair):
    """Our slots and upstream's namespaces differ by a handful of derived
    fields; the converter has to supply exactly those and no fewer."""
    from pulserver.pypulseq._events import as_namespace

    ours, theirs = pair
    for key in ("gx", "rf", "adc", "trig"):
        got = vars(as_namespace(ours[key]))
        want = vars(theirs[key])
        missing = set(want) - set(got)
        assert not missing, f"{key} is missing {sorted(missing)} that upstream reads"


def test_as_namespace_leaves_unset_registration_attributes_unset(pair):
    """``hasattr(event, 'shape_IDs')`` is load-bearing -- upstream branches on
    it, and an always-present one would make it trust ids from another
    sequence."""
    from pulserver.pypulseq._events import as_namespace

    ours, _ = pair
    assert not hasattr(as_namespace(ours["gx"]), "shape_IDs")
    assert not hasattr(as_namespace(ours["gx"]), "id")


def test_as_namespace_passes_through_anything_that_is_not_an_event():
    from pulserver.pypulseq._events import as_namespace

    for value in (3.5, "x", None, np.arange(4)):
        assert as_namespace(value) is value


def test_a_trigger_gets_the_channel_name_upstream_expects(pair):
    """Our slots keep the numeric code the file format carries."""
    from pulserver.pypulseq._events import as_namespace

    ours, theirs = pair
    assert as_namespace(ours["trig"]).channel == theirs["trig"].channel == "physio1"


def test_the_namespace_is_still_complete(pair):
    """Wrapping must not drop a name -- the drop-in promise is that this import
    covers everything ``import pypulseq`` would."""
    missing = {n for n in dir(upstream) if not n.startswith("_")} - set(pp.__all__)
    assert missing <= set(pp._EXCLUDED_UPSTREAM)


def test_a_wrapped_helper_keeps_its_upstream_signature():
    import inspect

    for name in ("calc_duration", "scale_grad", "split_gradient_at", "add_gradients"):
        assert inspect.signature(getattr(pp, name)) == inspect.signature(
            getattr(upstream, name)
        ), name
