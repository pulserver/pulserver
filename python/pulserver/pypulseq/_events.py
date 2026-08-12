"""PyPulseq's event factories, returning events with their fields in slots.

Every ``make_*`` here is upstream's, wrapped: it builds the event exactly as
PyPulseq does -- same validation, same defaults, same bug fixes when upstream
ships them -- and the result is converted once into a C++ object whose fields
are offsets rather than dictionary entries.

That conversion is the whole trick. Reading a ``SimpleNamespace`` costs a
dictionary lookup per field, which is unremarkable until a three-dimensional
protocol reads tens of millions of them; on a 512x1024x512 MPRAGE it is about
half the time spent building the sequence.

The objects that come back quack like the namespaces they replace. ``rf.signal``
is the complex waveform at its real amplitude and ``grad.waveform`` the scaled
samples, both rebuilt on demand, so anything downstream that reads an event --
including PyPulseq's own plotting and k-space code -- keeps working.

**Scaling is one number.** RF and arbitrary gradients are stored as a
normalised shape beside a scalar amplitude, which is how the file stores them
too. So::

    rf.amplitude *= 0.5          # one write; the registered shape still stands

is not merely faster than rescaling the samples, it is *better*: a variable
flip angle train is one magnitude shape at many amplitudes, and this is what
lets it be registered once. Assigning to ``signal`` or ``waveform`` outright
re-normalises and drops the registration, because that really is a new
waveform.

Setters do not re-validate. ``make_*`` checked the event when it built it, and
a loop moving a phase encode from one line to the next is not making a new
claim about the hardware.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, Callable

import numpy as _np
import pypulseq as _pp

from .._ext import _pulseqpp_wrapper as _cxx

# Only the factories are exported: SLOTTED is built from this list and is
# documented as "upstream's factories, wrapped". ``as_namespace`` and
# ``interoperating`` are the machinery underneath and stay module-private.
__all__ = [
    "make_adc",
    "make_adiabatic_pulse",
    "make_arbitrary_grad",
    "make_arbitrary_rf",
    "make_block_pulse",
    "make_delay",
    "make_digital_output_pulse",
    "make_extended_trapezoid",
    "make_gauss_pulse",
    "make_sinc_pulse",
    "make_soft_delay",
    "make_trapezoid",
    "make_trigger",
]


#: Which converter each PyPulseq event type goes through.
_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "rf": _cxx._rf_from,
    "trap": _cxx._trap_from,
    "grad": _cxx._grad_from,
    "adc": _cxx._adc_from,
    "labelset": _cxx._label_from,
    "labelinc": _cxx._label_from,
    "trigger": _cxx._trigger_from,
    "output": _cxx._trigger_from,
    "rot3D": _cxx._rotation_from,
    "soft_delay": _cxx._soft_delay_from,
    "delay": _cxx._delay_from,
}


def convert(event: Any) -> Any:
    """One PyPulseq event as a slotted one; anything else is passed through.

    Idempotent, so converting an already-converted event is free -- which
    matters because a factory that returns several events (a slice-selective
    pulse returns three) is converted element by element and callers may hand
    the results back in.
    """
    kind = getattr(event, "type", None)
    if kind is None or isinstance(event, _cxx.Event):
        return event
    convert_one = _CONVERTERS.get(kind)
    return event if convert_one is None else convert_one(event)


#: Fields upstream reads off an event that our slots do not carry under that
#: name, per event type: how to compute each from the event we do have.
#:
#: These are the whole difference between the two representations. A trapezoid
#: is flat-topped, so its endpoints are zero and we never stored them; an
#: arbitrary gradient keeps a normalised shape beside an amplitude, so its
#: area and duration are derived rather than held; and a trigger stores the
#: numeric channel code the file format uses rather than the name.
_COMPLETIONS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "trap": {
        "first": lambda e: 0.0,
        "last": lambda e: 0.0,
    },
    "grad": {
        "area": lambda e: float(_np.trapezoid(e.waveform, e.tt)),
        "shape_dur": lambda e: float(e.tt[-1] + (e.tt[-1] - e.tt[-2]) / 2)
        if len(e.tt) > 1
        else 0.0,
    },
    "trigger": {"channel": lambda e: _trigger_channel(e)},
    "output": {"channel": lambda e: _trigger_channel(e)},
}

#: Pulseq's trigger numbering as ``(control, channel_code) -> name``.
_TRIGGER_NAMES = {
    (1.0, 1.0): "osc0",
    (1.0, 2.0): "osc1",
    (1.0, 3.0): "ext1",
    (2.0, 1.0): "physio1",
    (2.0, 2.0): "physio2",
}


def _trigger_channel(event: Any) -> str:
    key = (float(getattr(event, "control", 0)), float(getattr(event, "channel_code", 0)))
    return _TRIGGER_NAMES.get(key, "")


def as_namespace(event: Any) -> Any:
    """One slotted event as the :class:`~types.SimpleNamespace` upstream reads.

    The inverse of :func:`convert`, and the reason the rest of PyPulseq's
    namespace works here at all. Upstream's helpers -- ``calc_duration``,
    ``align``, ``split_gradient``, ``scale_grad``, ``rotate`` -- do not merely
    read attributes off what they are given; they run ``isinstance(x,
    SimpleNamespace)`` checks and ``copy.deepcopy``, and a C++ event satisfies
    neither. Handing them a namespace is what makes those functions usable
    without forking any of them.

    Anything that is not one of our events is returned unchanged, so this is
    safe to map over an arbitrary argument list.
    """
    if not isinstance(event, _cxx.Event):
        return event

    fields = {}
    for name in dir(event):
        if name.startswith("_") or name in ("id", "shape_IDs"):
            continue
        try:
            fields[name] = getattr(event, name)
        except AttributeError:
            # Py_T_OBJECT_EX fields that were never assigned. Their *absence*
            # is load-bearing -- upstream branches on hasattr(event,
            # 'shape_IDs') -- so an unset one must stay unset.
            continue

    for name, derive in _COMPLETIONS.get(fields.get("type", ""), {}).items():
        if name not in fields:
            try:
                fields[name] = derive(event)
            except Exception:  # pragma: no cover - a shape too short to derive from
                pass

    made = _SimpleNamespace(**fields)
    # Carry the registration across when it exists, so a pre-registered event
    # keeps its ids through a helper that only reshapes it.
    for name in ("id", "shape_IDs"):
        if hasattr(event, name):
            setattr(made, name, getattr(event, name))
    return made


def _lowered(value: Any) -> Any:
    """``value`` with every event in it turned into a namespace, recursively."""
    if isinstance(value, _cxx.Event):
        return as_namespace(value)
    if isinstance(value, list):
        return [_lowered(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_lowered(item) for item in value)
    if isinstance(value, dict):
        return {key: _lowered(item) for key, item in value.items()}
    return value


def _raised(value: Any) -> Any:
    """``value`` with every namespace in it turned into an event, recursively."""
    if isinstance(value, list):
        return [_raised(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_raised(item) for item in value)
    return convert(value)


def interoperating(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function``, speaking namespaces inward and slotted events outward.

    One decorator for the whole of PyPulseq's namespace, not just its
    factories. Both directions are needed and for different reasons: outward,
    so what a caller gets back is the fast representation the rest of this
    package expects; inward, so upstream's own type checks and ``deepcopy``
    calls see what they were written against.

    Both conversions are identity on anything that is not an event, so a
    function that never touches one -- ``calc_ramp``, ``traj_to_grad`` -- pays
    only the walk over its arguments and is otherwise untouched.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        lowered_args = tuple(_lowered(value) for value in args)
        lowered_kwargs = {key: _lowered(value) for key, value in kwargs.items()}
        return _raised(function(*lowered_args, **lowered_kwargs))

    wrapper.__doc__ = (
        f"{function.__doc__ or ''}\n\n"
        "    Notes\n"
        "    -----\n"
        "    This is PyPulseq's function. Events go in as the namespaces it\n"
        "    expects and come back with their fields in slots. See\n"
        "    :mod:`pulserver.pypulseq._events`.\n"
    )
    return wrapper


def _converting(factory: Callable[..., Any]) -> Callable[..., Any]:
    """``factory``, with whatever it returns converted.

    Tuples are converted element by element: ``make_sinc_pulse(return_gz=True)``
    hands back the pulse and its two gradients together.

    Kept separate from :func:`interoperating` because a factory builds an event
    out of numbers rather than out of other events, so lowering its arguments
    would be a walk that never finds anything.
    """

    @functools.wraps(factory)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        made = factory(*args, **kwargs)
        if isinstance(made, tuple):
            return tuple(convert(one) for one in made)
        return convert(made)

    wrapper.__doc__ = (
        f"{factory.__doc__ or ''}\n\n"
        "    Notes\n"
        "    -----\n"
        "    This is PyPulseq's factory; the event it builds is returned with\n"
        "    its fields in slots. See :mod:`pulserver.pypulseq._events`.\n"
    )
    return wrapper


make_adc = _converting(_pp.make_adc)
make_adiabatic_pulse = _converting(_pp.make_adiabatic_pulse)
make_arbitrary_grad = _converting(_pp.make_arbitrary_grad)
make_arbitrary_rf = _converting(_pp.make_arbitrary_rf)
make_block_pulse = _converting(_pp.make_block_pulse)
make_delay = _converting(_pp.make_delay)
make_digital_output_pulse = _converting(_pp.make_digital_output_pulse)
make_extended_trapezoid = _converting(_pp.make_extended_trapezoid)
make_gauss_pulse = _converting(_pp.make_gauss_pulse)
make_sinc_pulse = _converting(_pp.make_sinc_pulse)
make_soft_delay = _converting(_pp.make_soft_delay)
make_trapezoid = _converting(_pp.make_trapezoid)
make_trigger = _converting(_pp.make_trigger)
