from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_epi_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "epi_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"epi_3d plugin not found: {plugin_path}")
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
        "nx", "ny", "nslices", "etl", "Rz", "bandwidth", "diffusion_bvalues", "diffusion_directions",
        "sequence_type", "user0_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "spin_echo"


def test_validation_reports_duration(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None


def test_validation_rejects_too_short_tr(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TR", 100.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


def test_validation_accepts_feasible_diffusion(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TE", 90.0)
    _set_value(protocol, "diffusion_bvalues", 800.0)

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is True


def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 4)  # partitions
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 2000.0)

    output_path = tmp_path / "epi_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "[BLOCKS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("ImagingMode") == "3d"
    assert seq.definitions.get("NumPartitions") == 4.0
    assert seq.definitions.get("NumShots") == 1.0

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3


def test_make_sequence_diffusion_adds_direction_repeats(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 16)
    _set_value(protocol, "ny", 8)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TE", 90.0)
    _set_value(protocol, "TR", 2000.0)
    _set_value(protocol, "diffusion_bvalues", 600.0)
    _set_value(protocol, "diffusion_directions", 3)

    output_path = tmp_path / "epi_3d_dwi_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    n_excitations = sum(
        1 for i in range(1, len(seq.block_events) + 1)
        if getattr(seq.get_block(i), "rf", None) is not None and seq.get_block(i).rf.use != "refocusing"
    )
    # 3 directions x 2 partitions x 1 shot = 6 excitations.
    assert n_excitations == 6
