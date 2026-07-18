from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest


def _load_plugin_module(plugin_path: Path):
    spec = importlib.util.spec_from_file_location("bridge_fse_2d_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "fse_2d.py"
    if not plugin_path.is_file():
        pytest.skip(f"fse_2d plugin not found: {plugin_path}")
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
        "nx", "ny", "nslices", "etl", "Ry", "bandwidth", "diffusion_bvalues", "diffusion_directions",
        "swap_phase_freq", "sequence_type", "user0_value",
    ]:
        assert key in protocol, f"missing key: {key}"

    assert protocol["sequence_type"]["value"] == "spin_echo"
    assert protocol["diffusion_bvalues"]["value"] == 0.0
    # Spin-echo role: the standard "flip" control drives the REFOCUSING
    # pulse (defaults to 180, prior hardcoded behavior) — the 90 excitation
    # is fixed, not user-selectable. Refocusing scheme defaults to constant
    # (not TRAPS/variable).
    assert protocol["flip"]["value"] == 180.0
    assert protocol["user0_value"]["value"] == 0.0


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
    _set_value(protocol, "TE", 3.0)

    result = plugin_module.validate_protocol(opts, protocol)

    assert result["valid"] is False
    assert result["duration"] is None


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


def test_validation_rejects_infeasible_diffusion(plugin_module):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "TE", 12.0)
    _set_value(protocol, "diffusion_bvalues", 2000.0)

    result = plugin_module.validate_protocol(opts, protocol)
    assert result["valid"] is False


def test_make_sequence_writes_real_pulseq_and_symmetric_k_space(plugin_module, tmp_path):
    """Regression test for a real bug: the post-readout rewind must return
    the NET per-echo x-moment to zero (reusing the -0.5*area prephaser), not
    do a "-full area" flyback rewind — every CPMG 180 flips the sign of the
    accumulated k_x, so a flyback rewind (designed for ordinary GRE echo
    trains with no intervening refocusing) drifts kx further positive every
    echo instead of returning to -kmax. Caught by checking kx is symmetric
    about zero, not just checking the k-space shape/size.
    """
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 2)
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 1000.0)

    output_path = tmp_path / "fse_2d_test.seq"
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
    kx_min, kx_max = k_adc[0].min(), k_adc[0].max()
    assert abs(kx_min + kx_max) < 1e-6 * max(abs(kx_min), abs(kx_max))
    assert kx_max > 0.4 * (32 / (220e-3))  # kx_max is half the full k-width (symmetric -kmax..+kmax)


def test_make_sequence_multishot_k_space_still_symmetric(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 4)  # multi-shot: 4 shots of 4 echoes each
    _set_value(protocol, "TR", 2000.0)

    output_path = tmp_path / "fse_2d_multishot_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "NumShots 4" in content

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    k_adc, *_ = seq.calculate_kspace()
    kx_min, kx_max = k_adc[0].min(), k_adc[0].max()
    assert abs(kx_min + kx_max) < 1e-6 * max(abs(kx_min), abs(kx_max))

    n_180 = sum(
        1 for i in range(1, len(seq.block_events) + 1)
        if getattr(seq.get_block(i), "rf", None) is not None and seq.get_block(i).rf.use == "refocusing"
    )
    assert n_180 == 16  # 4 shots x 4 echoes


def test_make_sequence_diffusion_flanks_only_first_180(plugin_module, tmp_path):
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 16)
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TE", 60.0)
    _set_value(protocol, "TR", 2000.0)
    _set_value(protocol, "diffusion_bvalues", 500.0)
    _set_value(protocol, "diffusion_directions", 3)

    output_path = tmp_path / "fse_2d_dwi_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    n_180 = sum(
        1 for i in range(1, len(seq.block_events) + 1)
        if getattr(seq.get_block(i), "rf", None) is not None and seq.get_block(i).rf.use == "refocusing"
    )
    # 3 directions x 1 slice x 16 echoes.
    assert n_180 == 48

    k_adc, *_ = seq.calculate_kspace()
    kx_min, kx_max = k_adc[0].min(), k_adc[0].max()
    assert abs(kx_min + kx_max) < 1e-6 * max(abs(kx_min), abs(kx_max))


def _refocusing_peak_amplitudes(seq) -> list[float]:
    peaks = []
    for i in range(1, len(seq.block_events) + 1):
        rf = getattr(seq.get_block(i), "rf", None)
        if rf is not None and rf.use == "refocusing":
            peaks.append(float(np.max(np.abs(rf.signal))))
    return peaks


def test_constant_nondefault_refocus_flip_scales_all_pulses(plugin_module, tmp_path):
    """Spin-echo role: the standard "flip" control drives the REFOCUSING
    pulse (90 excitation is fixed) — with the constant scheme (default),
    every 180 in the train should actually be scaled to the selected angle,
    not hardcoded to 180."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 8)  # == etl: exactly one shot
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TR", 2000.0)
    _set_value(protocol, "flip", 120.0)  # refocusing flip angle (deg)

    output_path = tmp_path / "fse_2d_refocus_const_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    peaks = _refocusing_peak_amplitudes(seq)
    assert len(peaks) == 8

    ratios = [p / peaks[0] for p in peaks]
    assert all(abs(r - 1.0) < 1e-6 for r in ratios)  # constant across the train

    # The peak amplitude must scale linearly with flip angle relative to a
    # freshly-built 180 deg reference pulse (same duration/time-bw/apod).
    fc = plugin_module.readout
    rf180_ref, _ = fc.build_refocusing_pulse(opts, 5.0e-3)
    ref_peak = float(np.max(np.abs(rf180_ref.signal)))
    assert abs(peaks[0] / ref_peak - 120.0 / 180.0) < 1e-6


def test_variable_refocus_scheme_ramps_toward_target(plugin_module, tmp_path):
    """Optional TSEplus-style TRAPS variable refocusing-angle scheme: echo 1
    starts above the target/plateau angle and decays monotonically toward
    it (never exceeding 180 deg)."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 32)
    _set_value(protocol, "ny", 8)  # == etl: exactly one shot
    _set_value(protocol, "nslices", 1)
    _set_value(protocol, "etl", 8)
    _set_value(protocol, "TR", 2000.0)
    _set_value(protocol, "flip", 120.0)  # target/plateau refocusing angle
    _set_value(protocol, "user0_value", 1.0)  # 1 = variable (TRAPS)

    output_path = tmp_path / "fse_2d_refocus_variable_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    peaks = _refocusing_peak_amplitudes(seq)
    assert len(peaks) == 8

    fc = plugin_module.readout
    rf180_ref, _ = fc.build_refocusing_pulse(opts, 5.0e-3)
    ref_peak = float(np.max(np.abs(rf180_ref.signal)))
    flip_deg = [180.0 * p / ref_peak for p in peaks]

    assert all(a - b > -1e-9 for a, b in zip(flip_deg, flip_deg[1:]))  # monotonically non-increasing
    assert flip_deg[0] > 120.0 + 1e-6  # starts above the plateau
    assert all(f <= 180.0 + 1e-6 for f in flip_deg)  # never exceeds 180
    assert abs(flip_deg[-1] - 120.0) < 5.0  # converges close to the target by the end of an 8-echo train


def test_gradient_surgery_reaches_shorter_esp_than_legacy(plugin_module):
    """The whole point of gradient surgery (fused crusher/prephase ramps
    that consume the full ESP/2 budget instead of a separate padding
    delay) is a strictly shorter minimum feasible echo spacing than the
    legacy (unfused, delay-padded) design, for the same geometry. This is
    a regression test for a real bug caught during development: an
    earlier draft that forced one shared ramp time across very different
    z/x amplitudes made the fused design WORSE than legacy instead of
    better (silently over-conservative, or outright slew-infeasible for
    the x-channel's prephase-to-readout jump)."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 256)
    _set_value(protocol, "ny", 256)
    _set_value(protocol, "nslices", 1)
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
    assert te_surgery < te_legacy


def test_surgery_sequence_passes_pypulseq_check_timing(plugin_module, tmp_path):
    """Block-continuity/timing self-consistency check (pypulseq's own
    validator) for the default (b_value == 0) gradient-surgery path —
    catches discontinuities between fused extended-trapezoid segments that
    ``calculate_kspace``'s symmetry check alone would not."""
    opts = pp.Opts()
    protocol = plugin_module.get_default_protocol(opts)
    _set_value(protocol, "nx", 64)
    _set_value(protocol, "ny", 64)
    _set_value(protocol, "nslices", 3)
    _set_value(protocol, "etl", 16)
    _set_value(protocol, "TR", 3000.0)

    output_path = tmp_path / "fse_2d_surgery_checktiming_test.seq"
    plugin_module.make_sequence(opts, protocol, str(output_path))

    seq = pp.Sequence(system=opts)
    seq.read(str(output_path))
    ok, err = seq.check_timing()
    assert ok, err
