from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_gre_mprage_radial_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre_mprage_radial_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre_mprage_radial_3d plugin not found: {plugin_path}")
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
        "TE", "TR", "Trecovery", "etl", "flip", "fov", "slice_thickness", "slice_spacing",
        "nx", "nslices", "num_shots", "Rz", "bandwidth", "sequence_type", "preparation_type",
        "user0_value", "user1_value", "user2_value", "user3_value", "user4_value", "user5_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["preparation_type"]["value"] == "inversion"
    assert protocol["sequence_type"]["value"] == "gradient_echo"


def test_validation_reports_duration_inversion(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None


def test_validation_reports_duration_t2prep(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "preparation_type", "t2_prep")

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None


def test_validation_with_mt_requires_longer_tr(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user5_value", 1.0)  # MT on, default TR too short

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is False

    _set_value(protocol, "TR", 20.0)
    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is True


def test_validation_rejects_too_short_ti(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user0_value", 0.01)

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is False


@pytest.mark.parametrize(
    ("prep_type", "expected_rf_use_counts"),
    [
        # 4 partitions x ceil(8 shots / etl=4) = 2 segments/partition = 8 segments total.
        ("inversion", {"i": 8}),
        ("t2_prep", {"p": 16, "r": 8}),
    ],
)
def test_make_sequence_prep_module_rf_counts(plugin_module, tmp_path, prep_type, expected_rf_use_counts):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "nslices", 4)  # partitions
    _set_value(protocol, "num_shots", 8)
    _set_value(protocol, "etl", 4)  # 2 segments per partition -> 4 total segments
    _set_value(protocol, "TR", 30.0)
    _set_value(protocol, "Trecovery", 200.0)
    _set_value(protocol, "preparation_type", prep_type)
    if prep_type == "inversion":
        _set_value(protocol, "user0_value", 300.0)
    else:
        _set_value(protocol, "user2_value", 40.0)

    output_path = tmp_path / "mprage_radial_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "[BLOCKS]" in content
    assert "extension ROTATIONS" in content
    assert re.search(r"^ImagingMode 3d\s*$", content, re.MULTILINE)
    assert re.search(r"^Trajectory radial\s*$", content, re.MULTILINE)
    assert re.search(rf"^PreparationType {prep_type}\s*$", content, re.MULTILINE)
    assert re.search(r"^NumPartitions 4\s*$", content, re.MULTILINE)

    rf_section = re.search(r"^\[RF\]\n(.*?)\n\n", content, re.MULTILINE | re.DOTALL)
    assert rf_section is not None
    use_chars = [line.split()[-1] for line in rf_section.group(1).strip().splitlines() if not line.startswith("#")]
    for use_char, expected_count in expected_rf_use_counts.items():
        assert use_chars.count(use_char) == expected_count, f"'{use_char}' count mismatch: {use_chars}"


def test_make_sequence_with_mt_adds_saturation_pulses(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "num_shots", 4)
    _set_value(protocol, "etl", 4)
    _set_value(protocol, "TR", 20.0)
    _set_value(protocol, "Trecovery", 200.0)
    _set_value(protocol, "user5_value", 1.0)

    output_path = tmp_path / "mprage_radial_3d_mt_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert re.search(r"^MTEnabled 1\s*$", content, re.MULTILINE)

    rf_section = re.search(r"^\[RF\]\n(.*?)\n\n", content, re.MULTILINE | re.DOTALL)
    use_chars = [line.split()[-1] for line in rf_section.group(1).strip().splitlines() if not line.startswith("#")]
    # 2 slices x 4 shots = 8 views, each preceded by one MT saturation pulse.
    assert use_chars.count("s") == 8
