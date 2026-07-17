from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_gre_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre_3d plugin not found: {plugin_path}")
    return _load_plugin_module(plugin_path)


def _set_value(protocol: dict, key: str, value) -> None:
    entry = protocol[key]
    entry["value"] = value
    if entry.get("type") == "stringlist":
        entry["index"] = entry["options"].index(value)


def test_plugin_contract(plugin_module):
    assert hasattr(plugin_module, "get_default_protocol")
    assert hasattr(plugin_module, "validate_protocol")
    assert hasattr(plugin_module, "make_sequence")


def test_default_protocol_has_expected_keys(plugin_module):
    protocol = plugin_module.get_default_protocol(pp.Opts())

    for key in [
        "TE", "TR", "flip", "fov", "phase_fov", "slice_thickness", "slice_spacing",
        "nx", "ny", "nslices", "Ry", "Rz", "bandwidth", "swap_phase_freq", "sequence_type",
    ]:
        assert key in protocol, f"missing key: {key}"

    # Single-echo file: no echo-train protocol surface at all.
    assert "num_echoes" not in protocol
    assert "imaging_mode" not in protocol
    assert "user0_value" not in protocol
    assert protocol["sequence_type"]["value"] == "gradient_echo"


def test_validation_reports_duration(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]


def test_validation_rejects_too_short_tr(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TR", 1.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 2)  # partitions
    _set_value(protocol, "TR", 30.0)

    output_path = tmp_path / "gre_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("ImagingMode") == "3d"
    assert seq.definitions.get("NumPartitions") == 2.0

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3
