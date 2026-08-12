"""Pulserver's drop-in replacement for :mod:`pypulseq`.

The complete public upstream namespace is re-exported unchanged, then
Pulserver's compatible replacements are layered on top, so a plugin needs one
import for the whole event layer::

    import pulserver.pypulseq as pp

    delay = pp.make_delay(1e-3)
    # A Sequence is built from the structure that repeats and how often, so its
    # every library row can be claimed before the first block is added.
    seq = pp.Sequence(pp.Opts(), 1, delay)

Only the objects listed in :data:`OVERRIDES` differ from upstream; everything
else *is* upstream, imported here so that ``import pypulseq`` alongside this
module is never necessary.

This namespace is the **event layer** only. The factories that build whole
sequence modules and scan loops — RF pulses, readouts, sampling plans, phase
schedules — live in :mod:`pulserver.design`::

    import pulserver.pypulseq as pp     # events, Sequence, Opts
    import pulserver.design as design  # modules and loops
"""

from __future__ import annotations

# ruff: noqa: I001

import pypulseq as _pypulseq

from . import _events
from ._make_label import COUNTER_LABELS, FLAG_LABELS, STICKY_FLAGS  # noqa: F401
from ._make_label import get_supported_labels as _get_supported_labels
from ._make_label import make_label as _make_label
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._opts import Opts as _Opts
from ._sequence import Sequence as _Sequence
from ._results import (
    AdcTimes as _AdcTimes,
    BTensor as _BTensor,
    DiffusionTable as _DiffusionTable,
    GradientSpectrum as _GradientSpectrum,
    KSpace as _KSpace,
    Pns as _Pns,
    RfPower as _RfPower,
    RfResponse as _RfResponse,
    RfTimes as _RfTimes,
    SoftDelay as _SoftDelay,
    Waveforms as _Waveforms,
    WaveformsAndTimes as _WaveformsAndTimes,
)
from ._simulate import bloch as _bloch
from ._simulate import sim_rf as _sim_rf
from ._transform_fov import TransformFOV as _TransformFOV

#: Upstream names Pulserver deliberately does not re-export: the adiabatic
#: factory (the preparation modules design their own), and the three shape
#: codec / unit helpers, which upstream exposes as modules rather than as
#: part of its authoring vocabulary.
_EXCLUDED_UPSTREAM = {
    "compress_shape",
    "convert",
    "decompress_shape",
    "make_adiabatic_pulse",
}

#: Upstream callables that must not be wrapped: they take or return no event,
#: and wrapping a class would replace its constructor with a plain function.
_UNWRAPPED_UPSTREAM = {"SigpyPulseOpts"}

# Upstream's own imports -- ``np``, ``math``, ``importlib``, the submodules its
# ``__init__`` happens to touch -- come across too. They are noise in a plugin's
# namespace, but ``import pulserver.pypulseq as pp`` promises to cover
# everything ``import pypulseq as pp`` would, and a contract test holds us to
# it. Completeness wins over tidiness; that is what drop-in means.
for _name in dir(_pypulseq):
    if _name.startswith("_") or _name in _EXCLUDED_UPSTREAM:
        continue
    _value = getattr(_pypulseq, _name)
    # Every callable in the namespace goes through the interoperation
    # decorator, not only the ``make_*`` factories. ``calc_duration(gx)``,
    # ``align(...)``, ``split_gradient(g)``, ``scale_grad(g, 2)`` and
    # ``rotate(...)`` all take events, and upstream implements them with
    # ``isinstance(x, SimpleNamespace)`` checks and ``copy.deepcopy`` -- both of
    # which a C++ event fails. Wrapping only the factories left the rest of the
    # namespace raising TypeError on our own events.
    if callable(_value) and not isinstance(_value, type) and _name not in _UNWRAPPED_UPSTREAM:
        _value = _events.interoperating(_value)
    globals()[_name] = _value
del _name, _value

Sequence = _Sequence
TransformFOV = _TransformFOV
AdcTimes = _AdcTimes
BTensor = _BTensor
DiffusionTable = _DiffusionTable
GradientSpectrum = _GradientSpectrum
KSpace = _KSpace
Pns = _Pns
RfPower = _RfPower
RfResponse = _RfResponse
RfTimes = _RfTimes
SoftDelay = _SoftDelay
Waveforms = _Waveforms
WaveformsAndTimes = _WaveformsAndTimes
Opts = _Opts
get_supported_labels = _get_supported_labels
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation
sim_rf = _sim_rf
bloch = _bloch

#: Upstream's factories, wrapped so the event they build comes back with its
#: fields in slots rather than in a dictionary.  Same validation, same
#: defaults, same bug fixes when upstream ships them -- see
#: :mod:`pulserver.pypulseq._events`.
SLOTTED = frozenset(_events.__all__) - _EXCLUDED_UPSTREAM

for _name in SLOTTED:
    globals()[_name] = getattr(_events, _name)
del _name

#: Everything in this namespace that is *not* upstream PyPulseq: Pulserver's
#: replacements for upstream objects, plus the extension events upstream has
#: no equivalent for.  This set is the single source of truth for both the
#: API reference page and the contract test that guards it.
#: What the analysis methods return under ``compat=False``.  Exported so a
#: caller can annotate or isinstance-check a result they were handed.
RESULTS = frozenset(
    {
        "AdcTimes",
        "BTensor",
        "DiffusionTable",
        "GradientSpectrum",
        "KSpace",
        "Pns",
        "RfPower",
        "RfResponse",
        "RfTimes",
        "SoftDelay",
        "Waveforms",
        "WaveformsAndTimes",
    }
)

OVERRIDES = frozenset(
    {
        *RESULTS,
        "COUNTER_LABELS",
        "FLAG_LABELS",
        "STICKY_FLAGS",
        "Opts",
        "Sequence",
        "TransformFOV",
        "get_supported_labels",
        "make_label",
        "make_rf_shim",
        "make_rotation",
        "sim_rf",
        "bloch",
        *SLOTTED,
    }
)

#: The upstream names re-exported verbatim.
UPSTREAM = (
    frozenset({_name for _name in dir(_pypulseq) if not _name.startswith("_")})
    - _EXCLUDED_UPSTREAM
    - OVERRIDES
)

__all__ = sorted(UPSTREAM | OVERRIDES)


#: Why each withheld upstream name is withheld. Reaching for one gets this
#: rather than a bare AttributeError, which would read as an oversight.
_WITHHELD_REASONS = {
    "make_adiabatic_pulse": (
        "not re-exported: the preparation modules in pulserver.design build their own "
        "adiabatic pulses, with the sweep and the spoiling the module needs. Import it "
        "from pypulseq directly if you want upstream's."
    ),
    "compress_shape": "a shape codec, not authoring vocabulary; import it from pypulseq.",
    "decompress_shape": "a shape codec, not authoring vocabulary; import it from pypulseq.",
    "convert": "a unit helper, not authoring vocabulary; import it from pypulseq.convert.",
}


def __getattr__(name: str):
    reason = _WITHHELD_REASONS.get(name)
    if reason is not None:
        raise AttributeError(f"pulserver.pypulseq does not export {name!r}: {reason}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
