"""Public PSD/design namespace contract."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pulserver
import pulserver.design as design
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver import ScanLoop, SequenceModule
from pulserver.design import (
    _lowlevel,
    calc_adc_timing,
    make_bssfp_readout,
    make_crusher,
    make_hard_pulse,
    make_line_readout,
    make_phase_blip,
    make_phase_cycling_schedule,
    make_phase_encoding,
    make_rf_spoiling_schedule,
    make_traps_schedule,
)
from pulserver.design._lowlevel import make_radial_tilt


def test_enhanced_pypulseq_contains_complete_upstream_namespace() -> None:
    withheld = {"compress_shape", "convert", "decompress_shape", "make_adiabatic_pulse"}
    upstream_names = {name for name in dir(upstream) if not name.startswith("_")} - withheld
    assert upstream_names <= set(pp.__all__)
    assert all(hasattr(pp, name) for name in upstream_names)
    assert all(not hasattr(pp, name) for name in withheld)
    assert issubclass(pp.Sequence, upstream.Sequence)
    assert pp.make_delay is upstream.make_delay
    assert callable(design.traj2grad)


def test_pypulseq_namespace_is_upstream_plus_a_declared_override_set() -> None:
    # The drop-in namespace is exactly upstream plus OVERRIDES -- nothing else.
    # That is what keeps its API reference page a short diff rather than a
    # duplicate of the upstream docs.
    assert set(pp.__all__) == set(pp.UPSTREAM) | set(pp.OVERRIDES)
    assert set(pp.UPSTREAM).isdisjoint(set(pp.OVERRIDES) - set(dir(upstream)))
    for name in pp.OVERRIDES:
        assert hasattr(pp, name)


def test_design_toolbox_is_disjoint_from_the_event_layer() -> None:
    # The split is by role: pypulseq builds events, design builds modules and
    # loops.  A name must never be reachable through both.
    assert set(design.__all__).isdisjoint(pp.__all__)
    assert set(_lowlevel.__all__).isdisjoint(pp.__all__)
    # The escape hatch is a strictly separate tier, never a second spelling of
    # a public factory.
    assert set(_lowlevel.__all__).isdisjoint(design.__all__)
    for name in design.__all__:
        assert not hasattr(pp, name), f"{name} leaked back into pulserver.pypulseq"


def test_design_factories_are_reachable_only_through_the_design_namespace() -> None:
    moved = {
        "make_line_readout",
        "make_slice_selective_pulse",
        "make_cartesian_sampling",
        "make_rf_spoiling_schedule",
        "make_crusher",
        "calc_adc_timing",
        "traj2grad",
    }
    assert moved <= set(design.__all__)
    for name in moved:
        assert callable(getattr(design, name))
        with pytest.raises(AttributeError):
            getattr(pp, name)


def test_pulserver_base_overrides_are_vendor_neutral_and_callable() -> None:
    system = pp.Opts()
    assert system.rf_dead_time == system.rf_ringdown_time == system.adc_dead_time == 0.0
    assert system.rf_raster_time == system.adc_raster_time == pytest.approx(2e-6)
    assert system.grad_raster_time == system.block_duration_raster == pytest.approx(10e-6)
    assert {"OFF", "MODULE"} <= set(pp.get_supported_labels())


def test_upstream_shape_codec_helpers_are_not_re_exported() -> None:
    """The shape codec and unit converter are not authoring vocabulary."""
    withheld = ("compress_shape", "convert", "decompress_shape", "make_adiabatic_pulse")
    for name in withheld:
        assert name not in pp.__all__
        assert name not in pp.UPSTREAM
        assert name not in pp.OVERRIDES
        with pytest.raises(AttributeError):
            getattr(pp, name)


def test_ordering_names_are_sequence_agnostic() -> None:
    generic = {
        "make_linear_order",
        "make_radial_order",
        "make_radial_adaptive_order",
        "make_shuffling_order",
    }
    old_fse_names = {name.replace("make_", "make_fse_") for name in generic}
    assert generic <= set(_lowlevel.__all__)
    assert old_fse_names.isdisjoint(_lowlevel.__all__)


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
        signature = inspect.signature(getattr(design, name))
        assert forbidden.isdisjoint(signature.parameters)
        assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def test_root_exports_plugin_contract_without_core_imports() -> None:
    assert pulserver.Sequence is pulserver.PulseqSequence
    assert callable(pulserver.run_cli)
    assert callable(pulserver.validate_protocol)
    assert isinstance(make_radial_tilt(4), ScanLoop)


def test_root_exports_the_abstract_authoring_types() -> None:
    assert pulserver.ScanLoop is ScanLoop
    assert pulserver.SequenceModule is SequenceModule
    assert {"ScanLoop", "SequenceModule"} <= set(pulserver.__all__)
    assert {"ScanLoop"}.isdisjoint(pp.__all__)
    # k-space and slices are one type: a loop is a table of positions plus a
    # grouping of them into shots, whatever the positions mean.
    assert {"SamplingPattern", "SliceSchedule", "SliceGroup"}.isdisjoint(pulserver.__all__)
    # The KSFoundation-style flattened plan is gone: outer loops are the
    # sequence's own, not a type the toolbox owns.
    assert {"Acquisition", "AcquisitionPlan"}.isdisjoint(pulserver.__all__)
    assert not hasattr(pp, "AcquisitionPlan")


def test_root_exports_the_encoding_axis_beside_the_loop_it_annotates() -> None:
    from pulserver.design._sampling import EncodingAxis as authoring_axis

    assert pulserver.EncodingAxis is authoring_axis
    assert "EncodingAxis" in pulserver.__all__
    # Abstract authoring types live in the root contract, not in either
    # authoring namespace -- the same rule ScanLoop follows.
    assert {"EncodingAxis"}.isdisjoint(pp.__all__)
    assert {"EncodingAxis"}.isdisjoint(design.__all__)


def test_label_set_partitions_into_loop_counters_and_module_flags() -> None:
    supported = set(pp.get_supported_labels())
    assert set(pp.COUNTER_LABELS) | set(pp.FLAG_LABELS) == supported
    assert set(pp.COUNTER_LABELS).isdisjoint(pp.FLAG_LABELS)
    # Counters are the ten ISMRMRD EncodingCounters fields the interpreter
    # tracks; everything else is a sticky block property.
    assert set(pp.COUNTER_LABELS) == {"LIN", "PAR", "SLC", "ECO", "PHS", "REP", "SET", "AVG", "SEG", "ACQ"}
    assert {"ONCE", "MODULE", "OFF", "NOROT", "NOPOS", "PMC"} <= set(pp.FLAG_LABELS)
    # Flags that outlive the module that sets them are never auto-reset.
    assert set(pp.STICKY_FLAGS) == {"ONCE", "MODULE", "TRID"}
    assert set(pp.STICKY_FLAGS) <= set(pp.FLAG_LABELS)


def test_every_counter_can_drive_a_scan_loop() -> None:
    # A loop is not a k-space object: any counter is a loop axis.
    for label in pp.COUNTER_LABELS:
        loop = design.make_counter_loop(3, label=label)
        assert isinstance(loop, ScanLoop)
        assert [event.label for event in loop.labels(1)] == [label]


def test_root_namespace_excludes_waveform_authoring_helpers() -> None:
    leaked = {"make_hard_pulse", "make_crusher", "make_line_readout", "make_radial_tilt"}
    assert leaked.isdisjoint(pulserver.__all__)
    for name in leaked:
        with pytest.raises(AttributeError):
            getattr(pulserver, name)
        assert hasattr(design, name) or hasattr(_lowlevel, name)


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
    assert "arbgrad" not in design.__all__
    # The raw waveform core is reachable, but only through the escape hatch.
    assert "arbgrad" in _lowlevel.__all__
    assert callable(design.traj2grad)


def test_implementation_namespaces_are_private() -> None:
    legacy_names = {"encoding", "readout", "rf", "sampling", "schedules", "system"}
    assert legacy_names.isdisjoint(pulserver.__all__)
    assert legacy_names.isdisjoint(pp.__all__)
    assert legacy_names.isdisjoint(design.__all__)
    assert all(not hasattr(pulserver, name) for name in legacy_names)
    assert all(not hasattr(pp, name) for name in legacy_names)
    assert all(not hasattr(design, name) for name in legacy_names)
    assert all(importlib.util.find_spec(f"pulserver.pypulseq.{name}") is None for name in legacy_names)
    assert all(importlib.util.find_spec(f"pulserver.design.{name}") is None for name in legacy_names)


def test_rf_and_readout_factories_return_only_common_module_protocol() -> None:
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    pulse = make_hard_pulse(np.deg2rad(10), duration=1e-3, system=system)
    readout = make_line_readout(system, (0.22, 0.22), (32, 32), spoil_position="none")

    assert isinstance(pulse, SequenceModule)
    assert isinstance(readout, SequenceModule)
    assert isinstance(readout.set_state(lin_idx=4), SequenceModule)
    excitation = design.make_slice_selective_pulse(np.deg2rad(25), 5e-3, duration=0.6e-3, system=system)
    bssfp = make_bssfp_readout(system, (0.22, 0.22), (32, 8), excitation)
    assert isinstance(bssfp, SequenceModule)


def test_gradient_helpers_use_physical_units_and_axes() -> None:
    system = pp.Opts()
    crusher = make_crusher(system, "z", dephasing_cycles=4.0, voxel_size=5e-3)
    explicit = make_crusher(system, "z", area=800.0)
    template = make_phase_encoding(system, "y", 0.24 / 64)
    blip = make_phase_blip(system, "y", 0.24, steps=2)

    assert crusher.area == pytest.approx(explicit.area)
    assert template.channel == "y" and template.area == pytest.approx(64 / (2 * 0.24))
    assert blip.area == pytest.approx(2 / 0.24)
    assert "make_spoiler" not in design.__all__


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


def _autosummary_entries(page: str) -> set[str]:
    text = (Path(__file__).resolve().parents[2] / "docs" / "api" / page).read_text()
    return {line.strip() for line in text.splitlines() if line.startswith("   pulserver.") and "(" not in line}


def test_every_design_factory_is_on_its_api_reference_page() -> None:
    # The split only pays for itself if the page stays a faithful index of the
    # namespace; a factory added without a doc entry is invisible to users.
    documented = _autosummary_entries("design.md")
    assert {f"pulserver.design.{name}" for name in design.__all__} == documented


def test_the_private_building_blocks_are_absent_from_the_public_surface() -> None:
    # The escape hatch is private: not on the reference page, not in __all__,
    # not reachable as an attribute of the public namespace.  Anything a plugin
    # genuinely needs is a gap in the factory layer, to be closed there.
    assert not hasattr(design, "lowlevel")
    assert not hasattr(design, "arbgrad")
    assert "lowlevel" not in design.__all__
    documented = _autosummary_entries("design.md")
    assert not any("_lowlevel" in entry for entry in documented)


def test_pypulseq_page_documents_the_overrides_and_nothing_upstream() -> None:
    # The point of the split: this page is a short diff against upstream, not a
    # second copy of the upstream reference.
    documented = _autosummary_entries("pypulseq.md")
    label_constants = {"COUNTER_LABELS", "FLAG_LABELS", "STICKY_FLAGS"}
    expected = {f"pulserver.pypulseq.{name}" for name in pp.OVERRIDES - label_constants}
    assert expected == documented
