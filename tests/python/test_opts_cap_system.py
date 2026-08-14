"""Capping a system's gradient and slew limits below what it reports.

The seqzoo plugins hold their sequences under a per-plugin ceiling with
:func:`cap_system`. These pin the contract that ceiling relies on: a copy is
returned so the offline CLI's shared system object is never mutated, the
limits are only ever lowered, and a ceiling above the hardware is a no-op.
"""

from __future__ import annotations

from pypulseq.convert import convert

from pulserver.pypulseq import Opts, cap_system


def _grad_mtm(opts):
    return convert(
        from_value=opts.max_grad, from_unit="Hz/m", to_unit="mT/m", gamma=opts.gamma
    )


def _slew_tms(opts):
    return convert(
        from_value=opts.max_slew, from_unit="Hz/m/s", to_unit="T/m/s", gamma=opts.gamma
    )


def test_cap_lowers_and_copies():
    system = Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    capped = cap_system(system, max_grad=40, max_slew=150)
    assert capped is not system
    assert round(_grad_mtm(capped)) == 40
    assert round(_slew_tms(capped)) == 150
    # The source is untouched, so a shared system object does not leak the cap.
    assert round(_grad_mtm(system)) == 80
    assert round(_slew_tms(system)) == 200


def test_cap_above_hardware_is_a_noop():
    system = Opts(max_grad=40, grad_unit="mT/m", max_slew=170, slew_unit="T/m/s")
    capped = cap_system(system, max_grad=80, max_slew=200)
    assert round(_grad_mtm(capped)) == 40
    assert round(_slew_tms(capped)) == 170


def test_none_leaves_an_axis_alone():
    system = Opts(max_grad=40, grad_unit="mT/m", max_slew=170, slew_unit="T/m/s")
    capped = cap_system(system, max_slew=120)
    assert round(_grad_mtm(capped)) == 40  # gradient untouched
    assert round(_slew_tms(capped)) == 120


def test_cap_is_idempotent():
    system = Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    once = cap_system(system, max_grad=30, max_slew=100)
    twice = cap_system(once, max_grad=30, max_slew=100)
    assert round(_grad_mtm(twice)) == round(_grad_mtm(once)) == 30
    assert round(_slew_tms(twice)) == round(_slew_tms(once)) == 100
