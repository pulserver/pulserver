"""Tests for SequenceCollection construction and basic accessors."""

import numpy as np
import pypulseq as pp
import pytest

from pge import Opts, SequenceCollection


def _safe_delay_collection() -> SequenceCollection:
    """Return a trivially safe collection for check() smoke tests."""
    seq = pp.Sequence(
        system=pp.Opts(
            max_grad=200,
            grad_unit="mT/m",
            max_slew=10000,
            slew_unit="T/m/s",
        )
    )
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(pp.make_delay(1e-3))
    return SequenceCollection(seq)


def test_from_sequence_object(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    assert sc.num_sequences == 1


def test_from_sequence_list(simple_gre_seq):
    sc = SequenceCollection([simple_gre_seq])
    assert sc.num_sequences == 1


def test_from_file(simple_gre_seq, tmp_path):
    seq_file = tmp_path / "test.seq"
    simple_gre_seq.write(str(seq_file))
    sc = SequenceCollection(str(seq_file))
    assert sc.num_sequences == 1


def test_invalid_type_raises():
    with pytest.raises(TypeError):
        SequenceCollection(42)


def test_num_blocks(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    n = sc.num_blocks(0)
    assert n > 0


def test_get_block(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    n = sc.num_blocks(0)

    block = None
    for idx in range(n):
        try:
            block = sc.get_block(0, idx)
            break
        except KeyError:
            continue

    assert block is not None


def test_get_sequence_copy(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    seq_copy = sc.get_sequence(0)
    assert isinstance(seq_copy, pp.Sequence)


def test_get_sequence_out_of_range(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    with pytest.raises(IndexError):
        sc.get_sequence(99)


def test_num_segments(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    ns = sc.num_segments(0)
    assert ns >= 1


def test_tr_size(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    ts = sc.tr_size(0)
    assert ts >= 1


def test_segment_size(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    ns = sc.num_segments(0)
    for i in range(ns):
        sz = sc.segment_size(0, i)
        assert sz >= 1


def test_report_returns_list(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    reps = sc.report()
    assert isinstance(reps, list)
    assert len(reps) >= 1


def test_report_fields(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    reps = sc.report()
    r = reps[0]
    assert hasattr(r, "num_blocks")
    assert hasattr(r, "segments")
    assert hasattr(r, "tr_size")
    assert hasattr(r, "tr_duration_s")
    assert r.tr_size >= 1
    assert r.tr_duration_s > 0


def test_report_print(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    text = sc.report(do_print=True)
    assert isinstance(text, str)
    assert "Subsequence" in text


def test_check_passes():
    sc = _safe_delay_collection()
    sc.check()


def test_check_with_pns():
    sc = _safe_delay_collection()
    sc.check(
        stim_threshold=1e9,
        decay_constant_us=360.0,
        pns_threshold_percent=1000.0,
    )


def test_sequence_collection_opts_override_system(simple_gre_seq):
    opts = Opts(
        gamma=float(simple_gre_seq.system.gamma),
        B0=float(simple_gre_seq.system.B0),
        max_grad=40.0,
        max_slew=150.0,
        b1_max_uT=10.0,
        rf_raster_time=float(simple_gre_seq.system.rf_raster_time),
        grad_raster_time=float(simple_gre_seq.system.grad_raster_time),
        adc_raster_time=float(simple_gre_seq.system.adc_raster_time),
        block_duration_raster=float(simple_gre_seq.system.block_duration_raster),
    )
    sc = SequenceCollection(simple_gre_seq, system=opts)
    assert sc.system is opts


def test_geopts_from_coil_model_defaults():
    opts = Opts.from_coil_model(
        "hrmw",
        gamma=42.576e6,
        B0=3.0,
        b1_max_uT=10.0,
    )
    assert np.isclose(opts.grad_raster_time, 4e-6)
    assert np.isclose(opts.rf_raster_time, 2e-6)
    assert np.isclose(opts.adc_raster_time, 2e-6)
    assert np.isclose(opts.chronaxie_us, 642.4)
    assert np.isclose(opts.default_stim_threshold(), 17.9 / 0.310)
    assert len(opts.forbidden_bands) > 0


def test_check_uses_opts_forbidden_bands_default(simple_gre_seq, monkeypatch):
    opts = Opts(
        gamma=float(simple_gre_seq.system.gamma),
        B0=float(simple_gre_seq.system.B0),
        max_grad=40.0,
        max_slew=150.0,
        forbidden_bands=[(100.0, 200.0, 1.0)],
        rf_raster_time=float(simple_gre_seq.system.rf_raster_time),
        grad_raster_time=float(simple_gre_seq.system.grad_raster_time),
        adc_raster_time=float(simple_gre_seq.system.adc_raster_time),
        block_duration_raster=float(simple_gre_seq.system.block_duration_raster),
    )
    sc = SequenceCollection(simple_gre_seq, system=opts)

    captured = {}

    def _fake_consistency(_cseq):
        return None

    def _fake_safety(_cseq, **kwargs):
        captured.update(kwargs)

    import pge.core._sequence as _sequence_mod

    monkeypatch.setattr(_sequence_mod, "_check_consistency", _fake_consistency)
    monkeypatch.setattr(_sequence_mod, "_check_safety", _fake_safety)

    sc.check()

    bands = captured["forbidden_bands"]
    assert len(bands) == 1
    assert np.isclose(bands[0][0], 100.0)
    assert np.isclose(bands[0][1], 200.0)
    assert np.isclose(bands[0][2], 1.0e-3 * float(opts.gamma))


# ── check() with fixture seq files ──────────────────────────────────


def test_check_passes_with_relaxed_opts(generated_seq_path, relaxed_check_opts):
    """check() passes for every fixture file when system limits are above actual peaks.

    Uses Opts(max_grad=50 mT/m, max_slew=350 T/m/s) which is above the
    effective peak values in all generated fixture sequences, so check()
    must not raise.
    """
    sc = SequenceCollection(str(generated_seq_path), system=relaxed_check_opts)
    sc.check()  # Must not raise


def test_check_fails_tight_slew(representative_generated_seq_path, relaxed_check_opts):
    """check() raises RuntimeError when max_slew is below the sequence peak slew.

    Uses max_slew=10 T/m/s which is far below the 150-200 T/m/s found in
    fixture sequences.  The C-backend additionally applies a vectorial
    safety factor (1/sqrt(3)), so the effective limit is ~5.8 T/m/s,
    ensuring failure for all representative fixtures.
    """
    tight = Opts(
        max_grad=50.0,   # mT/m — keep grad limit relaxed
        max_slew=10.0,   # T/m/s — far too tight; fixtures use 150-200 T/m/s
        B0=3.0,
        grad_raster_time=20e-6,
        block_duration_raster=20e-6,
    )
    sc = SequenceCollection(str(representative_generated_seq_path), system=tight)
    with pytest.raises(RuntimeError, match="slew"):
        sc.check()


def test_check_fails_tight_grad(representative_generated_seq_path, relaxed_check_opts):
    """check() raises RuntimeError when max_grad is below the sequence peak gradient.

    Uses max_grad=1 mT/m which is far below the ~35 mT/m found in fixture
    sequences, ensuring failure for all representative fixtures.
    """
    tight = Opts(
        max_grad=1.0,    # mT/m — far too tight; fixtures use up to ~35 mT/m
        max_slew=350.0,  # T/m/s — keep slew limit relaxed
        B0=3.0,
        grad_raster_time=20e-6,
        block_duration_raster=20e-6,
    )
    sc = SequenceCollection(str(representative_generated_seq_path), system=tight)
    with pytest.raises(RuntimeError, match="gradient"):
        sc.check()


def test_check_with_pns_passes_relaxed(representative_generated_seq_path, relaxed_check_opts):
    """check() with PNS params passes when stim_threshold is very high (no stimulation possible)."""
    sc = SequenceCollection(str(representative_generated_seq_path), system=relaxed_check_opts)
    sc.check(
        stim_threshold=1e9,   # Hz/m/s — effectively disables PNS failure
        decay_constant_us=360.0,
        pns_threshold_percent=1000.0,
    )


def test_construction_fails_incompatible_raster(representative_generated_seq_path):
    """SequenceCollection() raises RuntimeError when raster times are not
    integer multiples of the sequence's raster (detected at construction,
    before check() is called).
    """
    # grad_raster_time=7e-6 does not evenly divide the fixture's 20 µs raster
    incompat = Opts(
        max_grad=50.0,
        max_slew=350.0,
        B0=3.0,
        grad_raster_time=7e-6,
        block_duration_raster=7e-6,
    )
    with pytest.raises(RuntimeError, match="raster"):
        SequenceCollection(str(representative_generated_seq_path), system=incompat)


# ── validate() tests ─────────────────────────────────────────────────


def test_validate_returns_bool(simple_gre_seq):
    """validate() returns a bool."""
    sc = SequenceCollection(simple_gre_seq)
    result = sc.validate()
    assert isinstance(result, bool)


def test_validate_passes_generated(validate_pass_seq_path):
    """validate() should return True for all generated fixture files."""
    sc = SequenceCollection(str(validate_pass_seq_path))
    result = sc.validate()
    assert result is True


def test_validate_passes_with_averages(validate_pass_seq_path, num_averages):
    """validate() should return True for fixture files with multiple averages."""
    sc = SequenceCollection(str(validate_pass_seq_path), num_averages=num_averages)
    result = sc.validate()
    assert result is True


def test_validate_no_xml_path(simple_gre_seq):
    """validate() no longer accepts xml_path parameter."""
    sc = SequenceCollection(simple_gre_seq)
    with pytest.raises(TypeError):
        sc.validate(xml_path="dummy.xml")  # type: ignore[call-arg]


# ── pns() tests ──────────────────────────────────────────────────────


def test_pns_raises_without_params(representative_generated_seq_path):
    """pns() raises ValueError when stim_threshold/decay_constant_us not provided."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    with pytest.raises(ValueError, match="stim_threshold"):
        sc.pns()


def test_pns_runs_with_params(representative_generated_seq_path):
    """pns() runs without error given explicit PNS parameters."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.pns(stim_threshold=1e9, decay_constant_us=360.0)


def test_pns_single_sequence_idx(representative_generated_seq_path):
    """pns() with sequence_idx=0 runs without error."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.pns(sequence_idx=0, stim_threshold=1e9, decay_constant_us=360.0)


def test_pns_invalid_sequence_idx(representative_generated_seq_path):
    """pns() raises ValueError for out-of-range sequence_idx."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    with pytest.raises(ValueError):
        sc.pns(sequence_idx=9999, stim_threshold=1e9, decay_constant_us=360.0)


def test_calculate_pns_alias(representative_generated_seq_path):
    """calculate_pns() is an alias for pns() and runs without error."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.calculate_pns(stim_threshold=1e9, decay_constant_us=360.0)


# ── grad_spectrum() tests ────────────────────────────────────────────


def test_grad_spectrum_runs(representative_generated_seq_path):
    """grad_spectrum() runs without error on representative fixture files."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.grad_spectrum()


def test_grad_spectrum_single_sequence_idx(representative_generated_seq_path):
    """grad_spectrum() with sequence_idx=0 runs without error."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.grad_spectrum(sequence_idx=0)


def test_grad_spectrum_with_forbidden_bands(representative_generated_seq_path):
    """grad_spectrum() with explicit forbidden_bands runs without error."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.grad_spectrum(forbidden_bands=[(100.0, 500.0, 1.0), (2000.0, 3000.0, 0.5)])


def test_calculate_gradient_spectrum_alias(representative_generated_seq_path):
    """calculate_gradient_spectrum() is an alias for grad_spectrum()."""
    sc = SequenceCollection(str(representative_generated_seq_path))
    sc.calculate_gradient_spectrum()
