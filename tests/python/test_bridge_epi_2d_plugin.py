from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_epi_2d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "epi_2d.py"
    if not plugin_path.is_file():
        pytest.skip(f"epi_2d plugin not found: {plugin_path}")
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
        "nx", "ny", "nslices", "etl", "bandwidth", "diffusion_bvalues", "diffusion_directions",
        "swap_phase_freq", "sequence_type", "user0_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "spin_echo"
    assert protocol["diffusion_bvalues"]["value"] == 0.0


def test_validation_reports_duration(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]


def test_validation_rejects_too_short_te(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TE", 5.0)

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


def test_validation_rejects_infeasible_diffusion(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TE", 25.0)
    _set_value(protocol, "diffusion_bvalues", 5000.0)

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is False


def test_make_sequence_writes_real_pulseq_and_correct_k_space(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 500.0)

    output_path = tmp_path / "epi_2d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "# Pulseq sequence file" in content
    assert "[BLOCKS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("ImagingMode") == "2d"
    assert seq.definitions.get("ETL") == 16.0
    assert seq.definitions.get("NumShots") == 1.0

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3
    # Full linear k-space sweep must actually span both edges (not collapse
    # to a thin band) on both the readout and blipped phase-encode axes.
    assert np.ptp(k_adc[0]) > 0.8 * (32 / (220e-3))
    assert np.ptp(k_adc[1]) > 0.8 * (16 / (220e-3))


def test_make_sequence_multishot_produces_multiple_shots(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 4)  # multi-shot: 4 shots of 4 lines each
    _set_value(protocol, "TR", 500.0)

    output_path = tmp_path / "epi_2d_multishot_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "NumShots 4" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    n_rf = sum(1 for i in range(1, len(seq.block_events) + 1) if getattr(seq.get_block(i), "rf", None) is not None)
    # 4 shots x 2 RF pulses (90 + 180) each.
    assert n_rf == 8
