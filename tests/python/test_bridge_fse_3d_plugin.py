from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_fse_3d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "fse_3d.py"
    if not plugin_path.is_file():
        pytest.skip(f"fse_3d plugin not found: {plugin_path}")
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
        "nx", "ny", "nslices", "etl", "Ry", "Rz", "bandwidth",
        "diffusion_bvalues", "diffusion_directions", "sequence_type", "user0_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "spin_echo"
    # Spin-echo role: "flip" drives refocusing (90 excitation is fixed).
    assert protocol["flip"]["value"] == 180.0
    assert protocol["user0_value"]["value"] == 0.0


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


def test_validation_accepts_feasible_diffusion(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TE", 60.0)
    _set_value(protocol, "diffusion_bvalues", 500.0)

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is True


def test_make_sequence_writes_real_pulseq_and_symmetric_k_space(plugin_module, tmp_path):
    """Same regression coverage as fse_2d.py's k-space symmetry test, plus a
    kz-per-echo-constancy check: kz is fixed for a whole shot (one
    partition) and must survive every 180's sign flip unperturbed, using the
    same symmetric prephase/rephase treatment as ky (see fse_3d.py
    docstring)."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 4)  # partitions
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 1000.0)

    output_path = tmp_path / "fse_3d_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "[BLOCKS]" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    assert len(seq.block_events) > 0
    assert seq.definitions.get("ImagingMode") == "3d"
    assert seq.definitions.get("NumPartitions") == 4.0

    k_adc, *_ = seq.calculate_kspace()
    assert k_adc.shape[0] == 3

    kx_min, kx_max = k_adc[0].min(), k_adc[0].max()
    assert abs(kx_min + kx_max) < 1e-6 * max(abs(kx_min), abs(kx_max))

    kz = k_adc[2]
    # Within each partition, kz must be constant ACROSS EVERY ECHO of the
    # train (nx=32 ADC samples per echo, so index per-echo — not
    # per-sample — when checking cross-echo constancy; a prior version of
    # this test checked kz[:16], which is only half of the FIRST echo's own
    # 32 samples and is constant trivially regardless of correctness. Real
    # regression: an earlier gradient-surgery draft's z-bridge padding
    # step silently changed the bridge's net area when the target duration
    # exceeded the shortest feasible one, corrupting kz by hundreds of
    # rad/m — caught by properly indexing this check).
    samples_per_echo = 32
    n_per_partition = samples_per_echo * 16
    for par_idx in range(4):
        partition_kz = kz[par_idx * n_per_partition : (par_idx + 1) * n_per_partition]
        per_echo_kz = partition_kz[::samples_per_echo]
        assert np.ptp(per_echo_kz) < 1e-3


def test_make_sequence_diffusion_adds_direction_repeats(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 16)
    _set_value(protocol, "ny", 8)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TE", 60.0)
    _set_value(protocol, "TR", 1000.0)
    _set_value(protocol, "diffusion_bvalues", 500.0)
    _set_value(protocol, "diffusion_directions", 3)

    output_path = tmp_path / "fse_3d_dwi_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    n_90 = sum(
        1 for i in range(1, len(seq.block_events) + 1)
        if getattr(seq.get_block(i), "rf", None) is not None and seq.get_block(i).rf.use != "refocusing"
    )
    # 3 directions x 2 partitions x 1 shot = 6 excitations.
    assert n_90 == 6


def _refocusing_peak_amplitudes(seq) -> list[float]:
    peaks = []
    for i in range(1, len(seq.block_events) + 1):
        rf = getattr(seq.get_block(i), "rf", None)
        if rf is not None and rf.use == "refocusing":
            peaks.append(float(np.max(np.abs(rf.signal))))
    return peaks


def test_constant_nondefault_refocus_flip_scales_all_pulses(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 16)
    _set_value(protocol, "ny", 8)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TR", 1000.0)
    _set_value(protocol, "flip", 120.0)  # refocusing flip angle (deg)

    output_path = tmp_path / "fse_3d_refocus_const_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    peaks = _refocusing_peak_amplitudes(seq)
    assert len(peaks) == 16  # 2 partitions x 8 echoes

    ratios = [p / peaks[0] for p in peaks]
    assert all(abs(r - 1.0) < 1e-6 for r in ratios)

    fc = plugin_module.readout
    rf180_ref, _ = fc.build_refocusing_pulse(opts, 160.0e-3)  # default slab thickness (UIParam.SLICE_THICKNESS)
    ref_peak = float(np.max(np.abs(rf180_ref.signal)))
    assert abs(peaks[0] / ref_peak - 120.0 / 180.0) < 1e-6


def test_variable_refocus_scheme_ramps_toward_target(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 16)
    _set_value(protocol, "ny", 8)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TR", 1000.0)
    _set_value(protocol, "flip", 120.0)  # target/plateau refocusing angle
    _set_value(protocol, "user0_value", 1.0)  # 1 = variable (TRAPS)

    output_path = tmp_path / "fse_3d_refocus_variable_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    peaks = _refocusing_peak_amplitudes(seq)
    assert len(peaks) == 8

    fc = plugin_module.readout
    rf180_ref, _ = fc.build_refocusing_pulse(opts, 160.0e-3)  # default slab thickness (UIParam.SLICE_THICKNESS)
    ref_peak = float(np.max(np.abs(rf180_ref.signal)))
    flip_deg = [180.0 * p / ref_peak for p in peaks]

    assert all(a - b > -1e-9 for a, b in zip(flip_deg, flip_deg[1:]))
    assert flip_deg[0] > 120.0 + 1e-6
    assert all(f <= 180.0 + 1e-6 for f in flip_deg)
    assert abs(flip_deg[-1] - 120.0) < 5.0


def test_gradient_surgery_reaches_shorter_or_equal_esp_than_legacy(plugin_module):
    """Same rationale as fse_2d.py's equivalent test — gradient surgery
    must not make ESP worse than the legacy design, even with the
    additional per-partition kz phase-encode folded into the z-bridge
    (see fc.build_kz_crusher_bridge)."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 128)
    _set_value(protocol, "ny", 128)
    _set_value(protocol, "nslices", 16)
    _set_value(protocol, "etl", 16)

    prot = plugin_module.dict_to_protocol(protocol)
    cfg = plugin_module._read_protocol(prot)

    def min_feasible_te_ms(compute_fn, lo=1.0, hi=60.0, tol=0.01):
        best = None
        for _ in range(40):
            mid = (lo + hi) / 2.0
            cfg.te_s = mid * 1e-3
            result = compute_fn(opts=pp.Opts(), cfg=cfg, strict=True)
            if result is not None:
                best = mid
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return best

    te_surgery = min_feasible_te_ms(plugin_module._compute_timing_surgery)
    te_legacy = min_feasible_te_ms(plugin_module._compute_timing_legacy)
    assert te_surgery is not None and te_legacy is not None
    assert te_surgery <= te_legacy + 0.02


def test_surgery_sequence_passes_pypulseq_check_timing(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 32)
    _set_value(protocol, "nslices", 4)
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 1000.0)

    output_path = tmp_path / "fse_3d_surgery_checktiming_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    ok, err = seq.check_timing()
    assert ok, err
