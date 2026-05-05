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
    workspace_root = Path(__file__).resolve().parents[3]
    plugin_path = (
        workspace_root
        / "pulserver-interpreter"
        / "tree"
        / "pulserver"
        / "sequences"
        / "src"
        / "sequence5.py"
    )
    if not plugin_path.is_file():
        pytest.skip(f"sequence5 plugin not found: {plugin_path}")
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
