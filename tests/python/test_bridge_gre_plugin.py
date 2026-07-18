from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_sequence5_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sequence5_plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre plugin not found: {plugin_path}")
    return _load_plugin_module(plugin_path)


def test_sequence5_plugin_contract(sequence5_plugin_module):
    assert hasattr(sequence5_plugin_module, "get_default_protocol")
    assert hasattr(sequence5_plugin_module, "validate_protocol")
    assert hasattr(sequence5_plugin_module, "make_sequence")


def test_sequence5_default_protocol_has_gre_keys(sequence5_plugin_module):
    protocol = sequence5_plugin_module.get_default_protocol(pp.Opts())

    for key in [
        "TE",
        "TR",
        "flip",
        "fov",
        "phase_fov",
        "slice_thickness",
        "slice_spacing",
        "nx",
        "ny",
        "nslices",
        "bandwidth",
        "Ry",
        "user0_value",
        "swap_phase_freq",
    ]:
        assert key in protocol

    assert protocol["TE"]["type"] == "float"
    assert protocol["TR"]["type"] == "float"
    assert protocol["TE"]["unit"] == "ms"
    assert protocol["TR"]["unit"] == "ms"
    assert protocol["bandwidth"]["unit"] == "Hz/px"


def test_sequence5_validation_reports_duration(sequence5_plugin_module):
    opts = pp.Opts()
    protocol = sequence5_plugin_module.get_default_protocol(opts)

    result = sequence5_plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]


def test_sequence5_make_sequence_writes_real_pulseq(sequence5_plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = sequence5_plugin_module.get_default_protocol(opts)
    output_path = tmp_path / "sequence5_test.seq"

    sequence5_plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[VERSION]" in content
    assert "[BLOCKS]" in content
    assert "[DEFINITIONS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0


def _set_value(protocol: dict, key: str, value) -> None:
    entry = protocol[key]
    entry["value"] = value
    if entry.get("type") == "stringlist":
        entry["index"] = entry["options"].index(value)


def test_sequence5_k_space_round_trip(sequence5_plugin_module, tmp_path):
    """K-space round-trip: symmetric kx coverage, one readout per sampled PE
    line, and the expected sampled-line count under Ry + ACS."""
    opts = pp.Opts()
    protocol = sequence5_plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 32)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "Ry", 2.0)
    _set_value(protocol, "user0_value", 8.0)
    _set_value(protocol, "TR", 250.0)

    output_path = tmp_path / "gre_kspace_test.seq"
    sequence5_plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))

    expected_lines = {i for i in range(32) if i % 2 == 0}
    start = 32 // 2 - 8 // 2
    expected_lines |= set(range(start, start + 8))
    assert seq.definitions.get("NySampled") == float(len(expected_lines))

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3
    assert k_adc.shape[1] == 32 * len(expected_lines)

    kx_min, kx_max = k_adc[0].min(), k_adc[0].max()
    assert abs(kx_min + kx_max) < 1e-3 * max(abs(kx_min), abs(kx_max))

    fov_pe_m = 220e-3
    delta_k = 1.0 / fov_pe_m
    ky_per_readout = k_adc[1].reshape(len(expected_lines), 32).mean(axis=1)
    expected_ky = sorted((i - 16) * delta_k for i in expected_lines)
    import numpy as np
    assert np.allclose(sorted(ky_per_readout), expected_ky, atol=0.05 * delta_k)
