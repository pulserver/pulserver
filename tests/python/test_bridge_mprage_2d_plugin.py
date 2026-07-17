from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_mprage_2d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "mprage_2d.py"
    if not plugin_path.is_file():
        pytest.skip(f"mprage_2d plugin not found: {plugin_path}")
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
        "TE", "TR", "Trecovery", "etl", "flip", "fov", "phase_fov", "slice_thickness", "slice_spacing",
        "nx", "ny", "nslices", "Ry", "bandwidth", "swap_phase_freq", "sequence_type",
        "user0_value", "user1_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "gradient_echo"
    assert "imaging_mode" not in protocol
    assert protocol["user0_value"]["unit"] == "ms"  # TI
    assert protocol["user1_value"]["options"] == [0.0, 1.0]  # inversion mode toggle


def test_validation_reports_duration(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]


def test_validation_rejects_too_short_ti(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user0_value", 0.01)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


def test_validation_rejects_too_short_tr(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TR", 1.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


def test_validation_rejects_negative_trecovery(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "Trecovery", -1.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False


@pytest.mark.parametrize("inv_mode,inv_value", [("hard", 0.0), ("adiabatic", 1.0)])
def test_make_sequence_writes_real_pulseq(plugin_module, tmp_path, inv_mode, inv_value):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 4)
    _set_value(protocol, "TR", 10.0)
    _set_value(protocol, "user0_value", 300.0)  # TI, ms
    _set_value(protocol, "Trecovery", 200.0)
    _set_value(protocol, "user1_value", inv_value)

    output_path = tmp_path / "mprage_2d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("ImagingMode") == "2d"
    assert seq.definitions.get("NumSlices") == 2.0
    assert seq.definitions.get("ETL") == 4.0
    assert seq.definitions.get("InversionMode") == inv_mode
    assert abs(seq.definitions.get("TI") - 0.3) < 1e-9
    assert abs(seq.definitions.get("Trecovery") - 0.2) < 1e-9

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3


def test_make_sequence_segments_pe_by_etl(plugin_module, tmp_path):
    """With ETL < NY, each slice must fire more than one inversion (one per
    segment) — verified by counting inversion RF blocks (``use == "inversion"``).
    """
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 16)
    _set_value(protocol, "ny", 8)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 4)  # 2 segments per slice
    _set_value(protocol, "TR", 10.0)
    _set_value(protocol, "user0_value", 100.0)
    _set_value(protocol, "Trecovery", 50.0)

    output_path = tmp_path / "mprage_2d_segmented.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))

    inversion_count = 0
    for i in range(1, len(seq.block_events) + 1):
        block = seq.get_block(i)
        rf = getattr(block, "rf", None)
        if rf is not None and getattr(rf, "use", None) == "inversion":
            inversion_count += 1

    assert inversion_count == 2
