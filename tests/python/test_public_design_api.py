"""Public PSD/design namespace contract."""

from __future__ import annotations

import importlib.util

import numpy as np
import pulserver
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver import Module
from pulserver.pypulseq import (
    SamplingPattern,
    calc_adc_timing,
    make_crusher,
    make_hard_pulse,
    make_line_readout,
    make_phase_blip,
    make_phase_cycling_schedule,
    make_phase_encoding,
    make_rf_spoiling_schedule,
    make_spoiler,
    make_traps_schedule,
    radial_2d,
)


def test_enhanced_pypulseq_contains_complete_upstream_namespace() -> None:
    upstream_names = {name for name in dir(upstream) if not name.startswith("_")}
    assert upstream_names <= set(pp.__all__)
    assert all(hasattr(pp, name) for name in upstream_names)
    assert issubclass(pp.Sequence, upstream.Sequence)
    assert pp.make_delay is upstream.make_delay
    assert callable(pp.traj2grad)


def test_root_exports_plugin_contract_without_core_imports() -> None:
    assert pulserver.Sequence is pulserver.PulseqSequence
    assert callable(pulserver.run_cli)
    assert callable(pulserver.set_protocol_value)
    assert isinstance(radial_2d(4), SamplingPattern)


def test_root_namespace_excludes_waveform_authoring_helpers() -> None:
    leaked = {"make_hard_pulse", "make_crusher", "make_line_readout", "radial_2d", "SamplingPattern"}
    assert leaked.isdisjoint(pulserver.__all__)
    for name in leaked:
        with pytest.raises(AttributeError):
            getattr(pulserver, name)
        assert hasattr(pp, name)


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

    assert isinstance(pulse, Module)
    assert isinstance(readout, Module)
    assert isinstance(readout.set_state(lin_idx=4), Module)


def test_gradient_helpers_use_physical_units_and_axes() -> None:
    system = pp.Opts()
    crusher = make_crusher(system, "z", dephasing_cycles=4.0, voxel_size=5e-3)
    explicit = make_crusher(system, "z", area=800.0)
    template, areas = make_phase_encoding(system, "y", 0.24, 64)
    blip = make_phase_blip(system, "y", 0.24, steps=2)
    spoilers = make_spoiler(system, 5e-3, dephasing_cycles=4.0)

    assert crusher.area == pytest.approx(explicit.area)
    assert template.channel == "y" and len(areas) == 64
    assert blip.area == pytest.approx(2 / 0.24)
    assert {gradient.channel for gradient in spoilers} == {"x", "y", "z"}


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
