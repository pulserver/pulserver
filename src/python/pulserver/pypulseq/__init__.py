"""Pulserver's drop-in replacement for :mod:`pypulseq`.

The complete public upstream namespace is re-exported unchanged, then
Pulserver's compatible replacements are layered on top, so a plugin needs one
import for the whole event layer::

    import pulserver.pypulseq as pp

    delay = pp.make_delay(1e-3)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(delay)

Of the names this module advertises, only those in ``OVERRIDES`` differ from
upstream; the rest *are* upstream, imported here so that ``import pypulseq``
alongside this module is never necessary.

This namespace is the **event layer** only. The factories that build whole
sequence modules and scan loops — RF pulses, readouts, sampling plans, phase
schedules — live in :mod:`pulserver.design`::

    import pulserver.pypulseq as pp     # events, Sequence, Opts
    import pulserver.design as design  # modules and loops
"""

from __future__ import annotations


import pypulseq as _pypulseq

import functools as _functools
import inspect as _inspect

from . import _events
from ._angles import (
    calc_golden_angles as _calc_golden_angles,
    calc_projection_shell as _calc_projection_shell,
    calc_raga_angles as _calc_raga_angles,
    calc_tiny_golden_angles as _calc_tiny_golden_angles,
    calc_uniform_angles as _calc_uniform_angles,
)
from ._gradients import concatenate_gradients as _concatenate_gradients
from ._gradients import make_crusher as _make_crusher
from ._epi import calc_epi_order as _calc_epi_order
from ._gradients import make_phase_blip as _make_phase_blip
from ._gradients import make_phase_encoding as _make_phase_encoding
from ._make_label import (  # noqa: F401
    COUNTER_LABELS,
    ENCODING_COUNTERS,
    FLAG_LABELS,
    FRAME_COUNTERS,
    MRD_COUNTERS,
    MRD_FLAGS,
    SCANNER_FLAGS,
    STICKY_FLAGS,
    canonical_label,
)
from ._masks import (
    calc_calibration_lines as _calc_calibration_lines,
    calc_sampled_lines as _calc_sampled_lines,
    calc_sampled_pairs as _calc_sampled_pairs,
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
from ._rf_pulses import make_2d_selective_pulse as _make_2d_selective_pulse
from ._rf_pulses import make_sigpy_pulse as _make_sigpy_pulse
from ._files import bands_to_hz_per_m as _bands_to_hz_per_m
from ._files import read as _read
from ._files import read_asc_bands as _read_asc_bands
from ._files import read_esp_bands as _read_esp_bands
from ._files import write as _write
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
from ._block import block_to_events as _block_to_events
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._opts import Opts as _Opts
from ._opts import apply_system_derates as _apply_system_derates
from ._opts import cap_system as _cap_system
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
from ._tile import tile as _tile
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

#: Names that resolve but are not advertised: ``pp.x`` answers for a script
#: written against upstream, while ``__all__`` and the API page show the
#: authoring vocabulary only. ``make_sigpy_pulse`` is upstream's spelling of
#: :func:`make_slr_pulse`, which designs the same filter from keyword
#: arguments and pulls in no sigpy, so its ``SigpyPulseOpts`` argument bundle
#: has nothing here to configure; ``eps`` is a raster comparison tolerance.
_UNADVERTISED_UPSTREAM = {
    "SigpyPulseOpts",
    "eps",
    "make_sigpy_pulse",
}

#: Upstream names that are modules rather than authoring vocabulary -- ``np``,
#: ``math``, ``importlib``, and the submodules upstream's ``__init__`` happens
#: to touch. They stay reachable, because ``import pulserver.pypulseq as pp``
#: promises that every ``pp.x`` an upstream script writes still resolves, and a
#: contract test holds us to it. They are kept out of :data:`UPSTREAM`, and so
#: out of ``__all__`` and the API page, because reachable and advertised are
#: different promises: a plugin author reading the namespace should see the
#: vocabulary, not the imports it was built from.
_UPSTREAM_MODULES: set[str] = set()

for _name in dir(_pypulseq):
    if _name.startswith("_") or _name in _EXCLUDED_UPSTREAM:
        continue
    _value = getattr(_pypulseq, _name)
    if _inspect.ismodule(_value):
        _UPSTREAM_MODULES.add(_name)
        globals()[_name] = _value
        continue
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


def _fast_scale_grad(_upstream):
    """``scale_grad`` that stays in the slotted form for the common case.

    The hot one: a phase-encode loop scales the same prewinder once per shot,
    and the generic interoperation decorator would convert the event to a
    namespace and back on every call. A slotted gradient with no system to
    re-check against is scaled in place of that; anything else is upstream's,
    unchanged.
    """

    _scaled = _events.scaled_gradient

    @_functools.wraps(_upstream)
    def scale_grad(grad, scale, system=None):
        if system is None:
            try:
                return _scaled(grad, scale)
            except TypeError:
                pass
        return _upstream(grad, scale, system=system)

    return scale_grad


scale_grad = _fast_scale_grad(globals()["scale_grad"])

Sequence = _Sequence
TransformFOV = _TransformFOV
tile = _tile
block_to_events = _block_to_events
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
cap_system = _cap_system
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
concatenate_gradients = _concatenate_gradients
make_crusher = _make_crusher
make_phase_blip = _make_phase_blip
make_phase_encoding = _make_phase_encoding
make_2d_selective_pulse = _make_2d_selective_pulse
make_sigpy_pulse = _make_sigpy_pulse
make_slr_pulse = _make_slr_pulse
read = _read
write = _write
read_asc_bands = _read_asc_bands
read_esp_bands = _read_esp_bands
bands_to_hz_per_m = _bands_to_hz_per_m
make_sms_pulse = _make_sms_pulse
make_spsp_pulse = _make_spsp_pulse
traj_to_grad = _traj_to_grad
calc_epi_order = _calc_epi_order
calc_golden_angles = _calc_golden_angles
calc_projection_shell = _calc_projection_shell
calc_raga_angles = _calc_raga_angles
calc_calibration_lines = _calc_calibration_lines
calc_sampled_lines = _calc_sampled_lines
calc_sampled_pairs = _calc_sampled_pairs
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
#: the event interoperation layer.
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
#: already uses; see the MATLAB-parity layer.
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
        "concatenate_gradients",
        "make_crusher",
        "make_phase_blip",
        "make_phase_encoding",
        "make_2d_selective_pulse",
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
        "calc_calibration_lines",
        "calc_epi_order",
        "calc_golden_angles",
        "calc_projection_shell",
        "calc_raga_angles",
        "calc_sampled_lines",
        "calc_sampled_pairs",
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
        "cap_system",
        "ceil_to_raster",
        "make_phase_cycling_schedule",
        "make_rf_spoiling_schedule",
        "make_traps_schedule",
        "quantize_readout_timing",
        "round_to_raster",
    }
)

#: Reading a sequence from a file and writing one to it, and the gradient
#: response bands a scanner ships beside its own.
FILES = frozenset(
    {
        "read",
        "write",
        "read_asc_bands",
        "read_esp_bands",
        "bands_to_hz_per_m",
    }
)

OVERRIDES = frozenset(
    {
        *RESULTS,
        *MATLAB_PARITY,
        *BASE_FACTORIES,
        *SAMPLING,
        *SYSTEM,
        *FILES,
        "COUNTER_LABELS",
        "ENCODING_COUNTERS",
        "FLAG_LABELS",
        "FRAME_COUNTERS",
        "MRD_COUNTERS",
        "MRD_FLAGS",
        "SCANNER_FLAGS",
        "STICKY_FLAGS",
        "Opts",
        "Sequence",
        "TransformFOV",
        "block_to_events",
        "tile",
        "canonical_label",
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
    - _UPSTREAM_MODULES
    - _UNADVERTISED_UPSTREAM
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
