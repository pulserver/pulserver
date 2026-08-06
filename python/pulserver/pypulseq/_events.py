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
from typing import Any, Callable

import pypulseq as _pp

from .._ext import _pulseqpp_wrapper as _cxx

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


def _converting(factory: Callable[..., Any]) -> Callable[..., Any]:
    """``factory``, with whatever it returns converted.

    Tuples are converted element by element: ``make_sinc_pulse(return_gz=True)``
    hands back the pulse and its two gradients together.
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
