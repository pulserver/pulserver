from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_gre_multiecho_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre_multiecho_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre_multiecho_3d plugin not found: {plugin_path}")
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
        "TE", "num_echoes", "TR", "flip", "fov", "phase_fov", "slice_thickness", "slice_spacing",
        "nx", "ny", "nslices", "Ry", "Rz", "bandwidth", "swap_phase_freq", "sequence_type",
        "user0_value", "user1_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "gradient_echo"
    assert "imaging_mode" not in protocol
    assert "user2_value" not in protocol  # no ACS slot for the 3D file


@pytest.mark.parametrize("flyback", [True, False])
def test_validation_reports_duration(plugin_module, flyback):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user1_value", 1.0 if flyback else 0.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]


def test_validation_rejects_too_short_tr(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TR", 5.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


@pytest.mark.parametrize("flyback", [True, False])
def test_make_sequence_echo_train(plugin_module, flyback, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "user1_value", 1.0 if flyback else 0.0)
    _set_value(protocol, "num_echoes", 2)
    _set_value(protocol, "user0_value", 4.0)  # echo spacing, ms
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 2)  # partitions
    _set_value(protocol, "TR", 40.0)

    output_path = tmp_path / "gre_multiecho_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("NumEchoes") == 2.0
    assert bool(seq.definitions.get("FlybackReadout")) == flyback
    assert seq.definitions.get("ImagingMode") == "3d"
    assert seq.definitions.get("NumPartitions") == 2.0

    # Confirm PAR labels and k-space calculation both work end to end.
    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3

    adc_times_ms = []
    t_s = 0.0
    for i in range(1, len(seq.block_events) + 1):
        block = seq.get_block(i)
        if getattr(block, "adc", None) is not None:
            adc = block.adc
            adc_times_ms.append((t_s + adc.delay + 0.5 * adc.num_samples * adc.dwell) * 1e3)
        t_s += seq.block_durations[i]

    in_train_spacing_ms = np.diff(adc_times_ms[:2])
    assert np.allclose(in_train_spacing_ms, 4.0, atol=1e-3)
