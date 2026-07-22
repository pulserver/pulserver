"""Public PSD/design namespace contract."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pulserver
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver import SequenceModule
from pulserver.pypulseq import (
    SamplingPattern,
    calc_adc_timing,
    make_bssfp_readout,
    make_crusher,
    make_hard_pulse,
    make_line_readout,
    make_phase_blip,
    make_phase_cycling_schedule,
    make_phase_encoding,
    make_radial_tilt,
    make_rf_spoiling_schedule,
    make_traps_schedule,
)


def test_enhanced_pypulseq_contains_complete_upstream_namespace() -> None:
    upstream_names = {name for name in dir(upstream) if not name.startswith("_")} - {"make_adiabatic_pulse"}
    assert upstream_names <= set(pp.__all__)
    assert all(hasattr(pp, name) for name in upstream_names)
    assert not hasattr(pp, "make_adiabatic_pulse")
    assert issubclass(pp.Sequence, upstream.Sequence)
    assert pp.make_delay is upstream.make_delay
    assert callable(pp.traj2grad)


def test_pulserver_base_overrides_are_vendor_neutral_and_callable() -> None:
    system = pp.Opts()
    assert system.rf_dead_time == system.rf_ringdown_time == system.adc_dead_time == 0.0
    assert system.rf_raster_time == system.adc_raster_time == pytest.approx(2e-6)
    assert system.grad_raster_time == system.block_duration_raster == pytest.approx(10e-6)
    assert {"OFF", "MODULE"} <= set(pp.get_supported_labels())
    assert all(callable(getattr(pp, name)) for name in ("compress_shape", "decompress_shape", "convert"))
    assert all(
        getattr(pp, name).__module__ == "pulserver.pypulseq"
        for name in ("compress_shape", "decompress_shape", "convert")
    )


def test_ordering_names_are_sequence_agnostic() -> None:
    generic = {
        "make_linear_order",
        "make_radial_order",
        "make_radial_adaptive_order",
        "make_shuffling_order",
    }
    old_fse_names = {name.replace("make_", "make_fse_") for name in generic}
    assert generic <= set(pp.__all__)
    assert old_fse_names.isdisjoint(pp.__all__)


def test_preparation_signatures_have_no_slice_selection_controls() -> None:
    names = {
        "make_inversion_pulse",
        "make_fat_saturation_pulse",
        "make_mt_pulse",
        "make_ihmt_pulse",
        "make_bloch_siegert_pulse",
        "make_t2prep_pulse",
        "make_t1t2_prep_pulse",
        "make_diffusion_prep",
    }
    forbidden = {"slice_thickness", "return_gz", "center_pos"}
    for name in names:
        signature = inspect.signature(getattr(pp, name))
        assert forbidden.isdisjoint(signature.parameters)
        assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def test_root_exports_plugin_contract_without_core_imports() -> None:
    assert pulserver.Sequence is pulserver.PulseqSequence
    assert callable(pulserver.run_cli)
    assert callable(pulserver.validate_protocol)
    assert isinstance(make_radial_tilt(4), SamplingPattern)


def test_root_exports_the_abstract_authoring_types() -> None:
    assert pulserver.SamplingPattern is SamplingPattern
    assert pulserver.SequenceModule is SequenceModule
    assert {"SamplingPattern", "SliceSampling", "SliceGroup", "SequenceModule"} <= set(pulserver.__all__)
    assert {"SamplingPattern", "SliceSampling", "SliceGroup"}.isdisjoint(pp.__all__)


def test_root_namespace_excludes_waveform_authoring_helpers() -> None:
    leaked = {"make_hard_pulse", "make_crusher", "make_line_readout", "make_radial_tilt"}
    assert leaked.isdisjoint(pulserver.__all__)
    for name in leaked:
        with pytest.raises(AttributeError):
            getattr(pulserver, name)
        assert hasattr(pp, name)


def test_protocol_helpers_kept_out_of_the_public_contract() -> None:
    internal = {
        "param_to_dict",
        "dict_to_param",
        "set_protocol_value",
        "validate_protocol_entry",
        "expected_param_kind",
        "enum_options",
    }
    assert internal.isdisjoint(pulserver.__all__)


def test_arbgrad_is_not_part_of_the_public_surface() -> None:
    assert "arbgrad" not in pp.__all__
    assert callable(pp.traj2grad)


def test_implementation_namespaces_are_private() -> None:
    legacy_names = {"encoding", "readout", "rf", "sampling", "schedules", "system"}
    assert legacy_names.isdisjoint(pulserver.__all__)
    assert legacy_names.isdisjoint(pp.__all__)
    assert all(not hasattr(pulserver, name) for name in legacy_names)
    assert all(not hasattr(pp, name) for name in legacy_names)
    assert all(importlib.util.find_spec(f"pulserver.pypulseq.{name}") is None for name in legacy_names)


def test_rf_and_readout_factories_return_only_common_module_protocol() -> None:
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    pulse = make_hard_pulse(np.deg2rad(10), duration=1e-3, system=system)
    readout = make_line_readout(system, (0.22, 0.22), (32, 32), spoil_position="none")

    assert isinstance(pulse, SequenceModule)
    assert isinstance(readout, SequenceModule)
    assert isinstance(readout.set_state(lin_idx=4), SequenceModule)
    excitation = pp.make_slice_selective_pulse(np.deg2rad(25), 5e-3, duration=0.6e-3, system=system)
    bssfp = make_bssfp_readout(system, (0.22, 0.22), (32, 8), excitation)
    assert isinstance(bssfp, SequenceModule)


def test_gradient_helpers_use_physical_units_and_axes() -> None:
    system = pp.Opts()
    crusher = make_crusher(system, "z", dephasing_cycles=4.0, voxel_size=5e-3)
    explicit = make_crusher(system, "z", area=800.0)
    template, areas = make_phase_encoding(system, "y", 0.24, 64)
    blip = make_phase_blip(system, "y", 0.24, steps=2)

    assert crusher.area == pytest.approx(explicit.area)
    assert template.channel == "y" and len(areas) == 64
    assert blip.area == pytest.approx(2 / 0.24)
    assert "make_spoiler" not in pp.__all__


def test_readout_axes_are_fixed_and_spoiling_uses_cycles() -> None:
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    readout = make_line_readout(system, (0.22, 0.22), (64, 64), spoil_cycles=8.0, derate=False)
    assert readout._spoil_factor == pytest.approx(8.0)
    with pytest.raises(TypeError, match="fixed to x/y/z"):
        make_line_readout(system, (0.22, 0.22), (64, 64), ro_axis="y")
    with pytest.raises(TypeError, match="spoil_cycles"):
        make_line_readout(system, (0.22, 0.22), (64, 64), spoil_factor=2.0)


def test_adc_timing_is_feasible_on_both_rasters() -> None:
    dwell, duration = calc_adc_timing(
        96,
        3.7e-6,
        grad_raster_time=10e-6,
        adc_raster_time=100e-9,
    )
    assert dwell / 100e-9 == pytest.approx(round(dwell / 100e-9))
    assert duration / 10e-6 == pytest.approx(round(duration / 10e-6))
    assert duration == pytest.approx(96 * dwell)


def test_public_phase_and_flip_schedules() -> None:
    spoil = make_rf_spoiling_schedule(4)
    cycle = make_phase_cycling_schedule(5, (0.0, np.pi / 2))
    traps = make_traps_schedule(4, np.deg2rad(120))

    assert np.rad2deg(spoil) == pytest.approx([0.0, 0.0, 117.0, 351.0])
    assert cycle == pytest.approx([0.0, np.pi / 2, 0.0, np.pi / 2, 0.0])
    assert len(traps) == 4 and traps[0] > traps[-1] > 0
