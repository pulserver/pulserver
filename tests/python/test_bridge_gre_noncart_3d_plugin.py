from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_gre_noncart_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre_noncart_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre_noncart_3d plugin not found: {plugin_path}")
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
        "TE", "TR", "flip", "fov", "slice_thickness", "slice_spacing",
        "nx", "nslices", "num_shots", "Rz", "sequence_type", "user0_value", "user1_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "gradient_echo"
    assert "bandwidth" not in protocol
    assert "Ry" not in protocol


@pytest.mark.parametrize("trajectory", ["spiral", "rosette"])
def test_validation_reports_duration(plugin_module, trajectory):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user0_value", 1.0 if trajectory == "rosette" else 0.0)

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


@pytest.mark.parametrize("trajectory", ["spiral", "rosette"])
def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path, trajectory):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "nslices", 4)  # partitions
    _set_value(protocol, "num_shots", 8)
    _set_value(protocol, "TR", 30.0)
    _set_value(protocol, "user0_value", 1.0 if trajectory == "rosette" else 0.0)

    output_path = tmp_path / "gre_noncart_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content
    assert "extension ROTATIONS" in content
    assert re.search(rf"^Trajectory {trajectory}\s*$", content, re.MULTILINE)
    assert re.search(r"^ImagingMode 3d\s*$", content, re.MULTILINE)
    assert re.search(r"^NumShots 8\s*$", content, re.MULTILINE)
    assert re.search(r"^NumPartitions 4\s*$", content, re.MULTILINE)

    # 1 rotated block (the arb-gradient readout) per shot per partition.
    rotation_rows = re.search(r"^extension ROTATIONS \d+\n((?:\d+ .*\n)+)", content, re.MULTILINE)
    assert rotation_rows is not None
    n_rotation_rows = len(rotation_rows.group(1).strip().splitlines())
    assert n_rotation_rows == 8 * 4 * 1
