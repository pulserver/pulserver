from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_zte_2d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "zte_2d.py"
    if not plugin_path.is_file():
        pytest.skip(f"zte_2d plugin not found: {plugin_path}")
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

    for key in ["TR", "flip", "fov", "nx", "num_shots", "sequence_type", "user0_value"]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "gradient_echo"
    assert "TE" not in protocol  # TE is hardware-fixed, not user-set
    assert "bandwidth" not in protocol


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
    _set_value(protocol, "TR", 0.01)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


@pytest.mark.parametrize("order_mode", ["uniform", "golden"])
def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path, order_mode):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "num_shots", 16)
    _set_value(protocol, "TR", 3.0)
    _set_value(protocol, "user0_value", 1.0 if order_mode == "golden" else 0.0)

    output_path = tmp_path / "zte_2d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content
    # pypulseq 1.5.0 cannot parse the ROTATIONS extension back, so verify
    # structurally (same convention as gre_radial_2d.py).
    assert "extension ROTATIONS" in content
    assert re.search(r"^Trajectory zte\s*$", content, re.MULTILINE)
    assert re.search(rf"^SpokeOrder {order_mode}\s*$", content, re.MULTILINE)
    assert re.search(r"^NumShots 16\s*$", content, re.MULTILINE)
    assert re.search(r"^Gap 0\s*$", content, re.MULTILINE)

    rotation_rows = re.search(r"^extension ROTATIONS \d+\n((?:\d+ .*\n)+)", content, re.MULTILINE)
    assert rotation_rows is not None
    n_rotation_rows = len(rotation_rows.group(1).strip().splitlines())
    assert n_rotation_rows == 16  # 1 rotated block (gx+rf+adc) per spoke
