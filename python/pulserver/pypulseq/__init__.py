"""Pulserver's drop-in replacement for :mod:`pypulseq`.

The complete public upstream namespace is re-exported unchanged, then
Pulserver's compatible replacements are layered on top, so a plugin needs one
import for the whole event layer::

    import pulserver.pypulseq as pp

    delay = pp.make_delay(1e-3)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(delay)

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
from ._angles import (
    calc_golden_angles as _calc_golden_angles,
    calc_raga_angles as _calc_raga_angles,
    calc_tiny_golden_angles as _calc_tiny_golden_angles,
    calc_uniform_angles as _calc_uniform_angles,
)
from ._gradients import make_crusher as _make_crusher
from ._gradients import make_phase_blip as _make_phase_blip
from ._gradients import make_phase_encoding as _make_phase_encoding
from ._make_label import COUNTER_LABELS, FLAG_LABELS, STICKY_FLAGS  # noqa: F401
from ._masks import (
    calc_sampled_lines as _calc_sampled_lines,
    make_caipirinha_mask as _make_caipirinha_mask,
    make_centric_order as _make_centric_order,
    make_linear_order as _make_linear_order,
    make_poisson_disc_mask as _make_poisson_disc_mask,
    make_radial_adaptive_order as _make_radial_adaptive_order,
    make_radial_order as _make_radial_order,
    make_random_mask as _make_random_mask,
    make_shuffling_order as _make_shuffling_order,
)
from ._ordering import calc_traversal_order as _calc_traversal_order
from ._sampling import make_uniform_mask as _make_uniform_mask
from ._make_label import get_supported_labels as _get_supported_labels
from ._make_label import make_label as _make_label
from ._matlab_parity import calc_rf_power as _calc_rf_power
from ._matlab_parity import get_supported_rf_use as _get_supported_rf_use
from ._matlab_parity import get_supported_rf_uses as _get_supported_rf_uses
from ._matlab_parity import make_hexagon_gradient_area as _make_hexagon_gradient_area
from ._matlab_parity import verify_file_signature as _verify_file_signature
from ._rf_pulses import make_sigpy_pulse as _make_sigpy_pulse
from ._rf_pulses import make_slr_pulse as _make_slr_pulse
from ._rf_pulses import make_sms_pulse as _make_sms_pulse
from ._rf_pulses import make_spsp_pulse as _make_spsp_pulse
from ._rotate3d import rotate3D as _rotate3D
from ._shapes import restore_additional_shape_samples as _restore_additional_shape_samples
from ._schedules import make_phase_cycling_schedule as _make_phase_cycling_schedule
from ._schedules import make_rf_spoiling_schedule as _make_rf_spoiling_schedule
from ._schedules import make_traps_schedule as _make_traps_schedule
from ._timing import calc_adc_timing as _calc_adc_timing
from ._timing import ceil_to_raster as _ceil_to_raster
from ._timing import quantize_readout_timing as _quantize_readout_timing
from ._timing import round_to_raster as _round_to_raster
from ._traj_to_grad import traj_to_grad as _traj_to_grad
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._opts import Opts as _Opts
from ._opts import apply_system_derates as _apply_system_derates
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
from ._simulate import calc_rf_bandwidth as _calc_rf_bandwidth
from ._simulate import sim_rf as _sim_rf
from ._transform_fov import TransformFOV as _TransformFOV

#: Upstream names Pulserver deliberately does not re-export: the three shape
#: codec / unit helpers, which upstream exposes as modules rather than as part
#: of its authoring vocabulary.
_EXCLUDED_UPSTREAM = {
    "compress_shape",
    "convert",
    "decompress_shape",
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
apply_system_derates = _apply_system_derates
ceil_to_raster = _ceil_to_raster
round_to_raster = _round_to_raster
quantize_readout_timing = _quantize_readout_timing
make_phase_cycling_schedule = _make_phase_cycling_schedule
make_rf_spoiling_schedule = _make_rf_spoiling_schedule
make_traps_schedule = _make_traps_schedule
get_supported_labels = _get_supported_labels
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation
sim_rf = _sim_rf
bloch = _bloch
calc_adc_timing = _calc_adc_timing
calc_rf_bandwidth = _calc_rf_bandwidth
calc_rf_power = _calc_rf_power
make_crusher = _make_crusher
make_phase_blip = _make_phase_blip
make_phase_encoding = _make_phase_encoding
make_sigpy_pulse = _make_sigpy_pulse
make_slr_pulse = _make_slr_pulse
make_sms_pulse = _make_sms_pulse
make_spsp_pulse = _make_spsp_pulse
traj_to_grad = _traj_to_grad
calc_golden_angles = _calc_golden_angles
calc_raga_angles = _calc_raga_angles
calc_sampled_lines = _calc_sampled_lines
calc_tiny_golden_angles = _calc_tiny_golden_angles
calc_traversal_order = _calc_traversal_order
calc_uniform_angles = _calc_uniform_angles
make_caipirinha_mask = _make_caipirinha_mask
make_centric_order = _make_centric_order
make_linear_order = _make_linear_order
make_poisson_disc_mask = _make_poisson_disc_mask
make_radial_adaptive_order = _make_radial_adaptive_order
make_radial_order = _make_radial_order
make_random_mask = _make_random_mask
make_shuffling_order = _make_shuffling_order
make_uniform_mask = _make_uniform_mask
get_supported_rf_use = _get_supported_rf_use
#: PyPulseq's own spelling of the same tuple, reachable but not in ``__all__``
#: -- upstream keeps it in a submodule rather than its top-level namespace.
get_supported_rf_uses = _get_supported_rf_uses
make_hexagon_gradient_area = _make_hexagon_gradient_area
restore_additional_shape_samples = _restore_additional_shape_samples
#: Through the same decorator upstream's namespace goes through: its body
#: builds with PyPulseq's own helpers, which want namespaces.
rotate3D = _events.interoperating(_rotate3D)
verify_file_signature = _verify_file_signature

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

#: Routines MATLAB Pulseq defines that upstream PyPulseq has never ported.
#: Present here so a script translated from MATLAB finds them under the name it
#: already uses; see :mod:`pulserver.pypulseq._matlab_parity`.
MATLAB_PARITY = frozenset(
    {
        "calc_rf_power",
        "get_supported_rf_use",
        "make_hexagon_gradient_area",
        "restore_additional_shape_samples",
        "rotate3D",
        "sim_rf",
        "verify_file_signature",
    }
)

#: Base factories that return an event or a plain array rather than a
#: :class:`~pulserver.SequenceModule`, which is what puts them in this
#: namespace and not in :mod:`pulserver.design`.
BASE_FACTORIES = frozenset(
    {
        "calc_adc_timing",
        "make_crusher",
        "make_phase_blip",
        "make_phase_encoding",
        "make_sigpy_pulse",
        "make_slr_pulse",
        "make_sms_pulse",
        "make_spsp_pulse",
        "traj_to_grad",
    }
)

#: Undersampling masks, view orderings and projection angles. Plain arrays a
#: scan loop indexes with, so they belong here rather than in
#: :mod:`pulserver.design`, which holds only modules.
SAMPLING = frozenset(
    {
        "calc_golden_angles",
        "calc_raga_angles",
        "calc_sampled_lines",
        "calc_tiny_golden_angles",
        "calc_traversal_order",
        "calc_uniform_angles",
        "make_caipirinha_mask",
        "make_centric_order",
        "make_linear_order",
        "make_poisson_disc_mask",
        "make_radial_adaptive_order",
        "make_radial_order",
        "make_random_mask",
        "make_shuffling_order",
        "make_uniform_mask",
    }
)

#: System-limit derating, raster rounding, and the per-repetition RF phase and
#: flip lists a scan loop indexes. All of them answer from an :class:`Opts` and
#: a count rather than from a sequence, which is what puts them here.
SYSTEM = frozenset(
    {
        "apply_system_derates",
        "ceil_to_raster",
        "make_phase_cycling_schedule",
        "make_rf_spoiling_schedule",
        "make_traps_schedule",
        "quantize_readout_timing",
        "round_to_raster",
    }
)

OVERRIDES = frozenset(
    {
        *RESULTS,
        *MATLAB_PARITY,
        *BASE_FACTORIES,
        *SAMPLING,
        *SYSTEM,
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
        "bloch",
        "calc_rf_bandwidth",
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


#: Why each absent name is absent. Reaching for one gets this rather than a
#: bare AttributeError, which would read as an oversight. Two kinds: upstream
#: names deliberately not re-exported, and MATLAB names with no counterpart
#: here because the capability arrives another way.
_WITHHELD_REASONS = {
    "compress_shape": "a shape codec, not authoring vocabulary; import it from pypulseq.",
    "decompress_shape": "a shape codec, not authoring vocabulary; import it from pypulseq.",
    "convert": "a unit helper, not authoring vocabulary; import it from pypulseq.convert.",
    "add_custom_label": (
        "there is nothing to register: make_label accepts any label string, write and "
        "read round-trip it by name, Sequence.evaluate_labels reports its value, and a "
        "label the interpreter does not know it ignores."
    ),
    "add_ramps": (
        "not ported: it needs a working calc_ramp, and pypulseq.calc_ramp raises for "
        "any ramp of more than zero intermediate points. Design the ramp with "
        "traj_to_grad instead, which solves the same problem under the same limits."
    ),
}
_WITHHELD_REASONS["addCustomLabel"] = _WITHHELD_REASONS["add_custom_label"]
_WITHHELD_REASONS["addRamps"] = _WITHHELD_REASONS["add_ramps"]


def __getattr__(name: str):
    reason = _WITHHELD_REASONS.get(name)
    if reason is not None:
        raise AttributeError(f"pulserver.pypulseq does not export {name!r}: {reason}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
