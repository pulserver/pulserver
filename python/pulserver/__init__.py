"""Pulserver package root.

The package re-exports the most common protocol-building types at the top level
so sequence plugins can be written with short imports.

Examples
--------
>>> from pulserver import Sequence, UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> UIParam.TR in protocol
True

Notes
-----
Use ``pulserver.pypulseq`` for PyPulseq-compatible sequence/event helpers.
Plugin contracts, reusable module factories, and sampling tools are re-exported
from this package root. Their implementation packages are private.

The authoring modules (``io`` and ``pypulseq``) require the *optional*
``pypulseq`` dependency, so they are imported lazily (PEP 562): ``import
pulserver`` — and hence ``pulserver.recon``, which runs in the scanner recon
env without ``pypulseq`` — stays import-clean; accessing an authoring name pulls
it in (raising a clear error if the extra is absent).
"""

from __future__ import annotations

import importlib

__all__ = [
    "pypulseq",
    "io",
    "params",
    "Sequence",
    "PulseqSequence",
    "Module",
    "run_cli",
    "UIParam",
    "Validate",
    "ParamKind",
    "InputMode",
    "FloatKey",
    "IntKey",
    "BoolKey",
    "EnumKey",
    "SequenceType",
    "ImagingMode",
    "PreparationType",
    "TriggerType",
    "TypeinFloatParam",
    "DropdownFloatParam",
    "TypeinIntParam",
    "DropdownIntParam",
    "BoolParam",
    "StringListParam",
    "Description",
    "Protocol",
    "ProtocolValue",
    "expected_param_kind",
    "enum_options",
    "make_enum_param",
    "validate_protocol_entry",
    "validate_protocol",
    "param_to_dict",
    "dict_to_param",
    "protocol_to_dict",
    "dict_to_protocol",
    "set_protocol_value",
    "calc_adc_timing",
    "make_crusher",
    "make_phase_encoding",
    "make_phase_blip",
    "make_spoiler",
    "make_rf_spoiling_schedule",
    "make_phase_cycling_schedule",
    "make_traps_schedule",
    "make_hard_pulse",
    "make_adiabatic_pulse",
    "make_sigpy_pulse",
    "make_slr_pulse",
    "make_frequency_selective_pulse",
    "make_slice_selective_pulse",
    "make_spsp_pulse",
    "make_spatially_selective_pulse",
    "make_spiral_selective_pulse",
    "make_2d_selective_pulse",
    "make_3d_selective_pulse",
    "make_inversion_pulse",
    "make_refocusing_pulse",
    "make_mt_pulse",
    "make_ihmt_pulse",
    "make_bloch_siegert_pulse",
    "make_t2prep_pulse",
    "make_t1t2_prep_pulse",
    "make_diffusion_prep",
    "make_fat_saturation_pulse",
    "make_line_readout",
    "make_epi_readout",
    "make_fse_readout",
    "make_radial_readout",
    "make_spiral_readout",
    "make_rosette_readout",
    "make_noncartesian_3d_readout",
    "make_zte_readout",
    "SamplingPattern",
    "SliceGroup",
    "chunk_indices",
    "linear_order",
    "outer_inner_order",
    "outer_product",
    "sampled_lines",
    "fse_linear_order",
    "fse_radial_order",
    "fse_radial_adaptive_order",
    "fse_shuffling_order",
    "random_mask",
    "caipirinha_mask",
    "poisson_disc_mask",
    "from_mask",
    "from_relative_shifts",
    "skipped_caipi",
    "radial_2d",
    "golden_angles",
    "uniform_angles",
    "golden_means_3d",
    "spiral_phyllotaxis",
    "directions_to_rotations",
    "slice_groups",
]


_PYPULSEQ_EXPORTS = {
    name
    for name in __all__
    if name.startswith("make_")
    or name
    in {
        "calc_adc_timing",
        "SamplingPattern",
        "SliceGroup",
        "chunk_indices",
        "linear_order",
        "outer_inner_order",
        "outer_product",
        "sampled_lines",
        "fse_linear_order",
        "fse_radial_order",
        "fse_radial_adaptive_order",
        "fse_shuffling_order",
        "random_mask",
        "caipirinha_mask",
        "poisson_disc_mask",
        "from_mask",
        "from_relative_shifts",
        "skipped_caipi",
        "radial_2d",
        "golden_angles",
        "uniform_angles",
        "golden_means_3d",
        "spiral_phyllotaxis",
        "directions_to_rotations",
        "slice_groups",
    }
}
_CORE_MODULES = {"params"}
_CORE_FUNCTIONS = {"make_enum_param"}


def __getattr__(name: str):
    if name in ("io", "pypulseq"):
        return importlib.import_module(f"{__name__}.{name}")
    if name in _CORE_MODULES:
        return importlib.import_module(f"{__name__}._core._protocol")
    if name in _CORE_FUNCTIONS:
        return getattr(importlib.import_module(f"{__name__}._core"), name)
    if name in _PYPULSEQ_EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.pypulseq"), name)
    if name in __all__:
        return getattr(importlib.import_module(f"{__name__}._core"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
