from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_gre_radial_2d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre_radial_2d.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre_radial_2d plugin not found: {plugin_path}")
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
        "nx", "nslices", "num_shots", "bandwidth", "sequence_type", "user0_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "gradient_echo"
    assert "phase_fov" not in protocol  # no separate PE dimension for radial
    assert "swap_phase_freq" not in protocol
    assert protocol["user0_value"]["options"] == [0.0, 1.0]  # spoke order toggle


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


@pytest.mark.parametrize("order_mode", ["uniform", "golden"])
def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path, order_mode):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "num_shots", 8)
    _set_value(protocol, "TR", 10.0)
    _set_value(protocol, "user0_value", 1.0 if order_mode == "golden" else 0.0)

    output_path = tmp_path / "gre_radial_2d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content
    # pypulseq 1.5.0 cannot parse the ROTATIONS extension back (see
    # pge-fixture-regeneration memory note), so verify structurally instead
    # of round-tripping through pp.Sequence().read().
    assert "extension ROTATIONS" in content
    assert re.search(r"^Trajectory radial\s*$", content, re.MULTILINE)
    assert re.search(rf"^SpokeOrder {order_mode}\s*$", content, re.MULTILINE)
    assert re.search(r"^NumShots 8\s*$", content, re.MULTILINE)
    assert re.search(r"^NumSlices 2\s*$", content, re.MULTILINE)

    # 3 rotated blocks (prephaser, readout, spoiler) per shot per slice.
    rotation_rows = re.search(r"^extension ROTATIONS \d+\n((?:\d+ .*\n)+)", content, re.MULTILINE)
    assert rotation_rows is not None
    n_rotation_rows = len(rotation_rows.group(1).strip().splitlines())
    assert n_rotation_rows == 8 * 2 * 3
