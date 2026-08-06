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

for _name in dir(_pypulseq):
    if not _name.startswith("_") and _name not in _EXCLUDED_UPSTREAM:
        globals()[_name] = getattr(_pypulseq, _name)
del _name

Sequence = _Sequence
Opts = _Opts
get_supported_labels = _get_supported_labels
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation

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
OVERRIDES = frozenset(
    {
        "COUNTER_LABELS",
        "FLAG_LABELS",
        "STICKY_FLAGS",
        "Opts",
        "Sequence",
        "get_supported_labels",
        "make_label",
        "make_rf_shim",
        "make_rotation",
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
