"""Lowering a system's gradient and slew limits below what it reports.

The sequence plugins hold their sequences under a per-plugin ceiling with
:func:`cap_system`, and a trajectory designer keeps headroom below that with
:func:`apply_system_derates`. These pin the contract both rely on: a copy is
returned so a shared system object is never mutated, the limits are only ever
lowered, and repeating either does not compound.
"""

from __future__ import annotations

from pypulseq.convert import convert

from pulserver.pypulseq import Opts, apply_system_derates, cap_system


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


def test_derating_copies_rather_than_mutating_the_callers_system():
    """A designer's headroom must not become the sequence's declared limit.

    The system object is shared: a plugin hands the same one to every module
    it builds and then to the ``Sequence`` itself. Derating it in place would
    lower the limits that events built *earlier* were designed against, and
    they would be over a ceiling that moved under them.
    """
    system = Opts(max_grad=40, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    derated = apply_system_derates(system)

    assert derated is not system
    assert derated.max_grad == 0.9 * system.max_grad
    assert derated.max_slew == 0.9 * system.max_slew
    assert round(_grad_mtm(system)) == 40
    assert round(_slew_tms(system)) == 200


def test_derating_is_idempotent():
    system = Opts(max_grad=40, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    once = apply_system_derates(system)
    twice = apply_system_derates(once)

    assert twice.max_grad == once.max_grad
    assert twice.max_slew == once.max_slew
