"""Unit tests for pulserver.design.readout.line (Line2D/Line3D)."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import readout

OPTS_KW = dict(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _opts():
    return pp.Opts(**OPTS_KW)


def _labels(seq, name):
    """(block_index, type, value) for every ``name`` label event in ``seq``."""
    out = []
    for i in range(1, seq.block_events.__len__() + 1):
        blk = seq.get_block(i)
        if blk.label is None:
            continue
        for lab in blk.label.values():
            if lab.label == name:
                out.append((i, lab.type, lab.value))
    return out


# ----------------------------------------------------------------------
# Balanced / unbalanced / SSFP-Echo net gradient moment.
# ----------------------------------------------------------------------


def test_balanced_single_echo_returns_to_zero() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="none")
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    _, k_full, *_ = seq.calculate_kspace()
    assert np.allclose(k_full[:, -1], 0.0, atol=1e-6)


def test_unbalanced_post_single_echo_leaves_target_spoil_moment() -> None:
    """gre.py convention: the post-spoiler's own area is spoil_delta alone,
    tacked onto wherever the (unmodified) natural echo excursion ends."""
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="post", spoil_factor=1.0)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    _, k_full, *_ = seq.calculate_kspace()
    expected_x = train._ro.n_post * train._ro.delta_k + train._spoil_factor * train._ro.k_width
    assert k_full[0, -1] == pytest.approx(expected_x, rel=1e-6)
    assert k_full[1, -1] == pytest.approx(0.0, abs=1e-6)  # PE always fully rewound


def test_unbalanced_pre_single_echo_leaves_target_spoil_moment() -> None:
    """Mirrors the post case: the pre-spoiler's own area is spoil_delta alone
    (signed to match the echo's plateau, giving a clean rise-then-settle
    staircase); the trailing rewind is the plain, spoil-independent formula,
    so the net moment left over is spoil_delta plus the prewind's own
    (unspoiled) magnitude -- see module docstring."""
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="pre", spoil_factor=1.0)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    _, k_full, *_ = seq.calculate_kspace()
    expected_x = train._ro.n_pre * train._ro.delta_k + train._spoil_factor * train._ro.k_width
    assert k_full[0, -1] == pytest.approx(expected_x, rel=1e-6)
    assert k_full[1, -1] == pytest.approx(0.0, abs=1e-6)  # PE always fully rewound


def test_pre_spoil_bridge_is_a_monotonic_staircase_no_reversal() -> None:
    """(0, spoil_plateau, readout_plateau) -- never dips through/past zero to
    reach the target area, unlike a naive combined position+spoil target."""
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="pre", spoil_factor=1.0)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    gx = seq.get_block(1).gx
    assert gx.type == "grad"
    assert np.all(gx.waveform >= 0.0)  # monotonic-ish staircase, same sign throughout
    assert gx.waveform[0] == pytest.approx(0.0)
    assert gx.waveform[-1] == pytest.approx(train._ro.gx.amplitude)


def test_rejects_bad_spoil_position() -> None:
    opts = _opts()
    with pytest.raises(ValueError):
        readout.Line2D(opts, (0.22, 0.22), (64, 64), spoil_position="bogus")


# ----------------------------------------------------------------------
# ADC k-window (symmetric echo and partial Fourier asymmetry).
# ----------------------------------------------------------------------


def test_symmetric_echo_adc_window_centered() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=32)  # ny/2 -> zero PE area
    k_adc, *_ = seq.calculate_kspace()
    half_span = 0.5 * train._ro.k_width - 0.5 * train._ro.delta_k
    assert k_adc[0, 0] == pytest.approx(-half_span, rel=1e-6)
    assert k_adc[0, -1] == pytest.approx(half_span, rel=1e-6)


def test_partial_fourier_truncates_pre_echo_side_only() -> None:
    opts = _opts()
    full = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, pf=1.0)
    part = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, pf=0.75)
    assert part.n_samples < full.n_samples
    assert part._ro.n_post == full._ro.n_post  # post-echo side untouched
    assert part._ro.n_pre < full._ro.n_pre  # pre-echo side truncated

    seq = pp.Sequence(opts)
    part(seq, pe_idx=32)
    k_adc, *_ = seq.calculate_kspace()
    assert k_adc[0, -1] == pytest.approx(0.5 * full._ro.k_width - 0.5 * full._ro.delta_k, rel=1e-6)


def test_pf_and_oversamp_reject_bad_values() -> None:
    opts = _opts()
    with pytest.raises(ValueError):
        readout.Line2D(opts, (0.22, 0.22), (64, 64), pf=0.3)
    with pytest.raises(ValueError):
        readout.Line2D(opts, (0.22, 0.22), (64, 64), oversamp=0.5)


# ----------------------------------------------------------------------
# Multi-echo: bipolar alternation and flyback retrace, ECO labels.
# ----------------------------------------------------------------------


def test_bipolar_echoes_alternate_and_retrace_same_window() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=4, flyback=False, spoil_position="none")
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    k_adc, *_ = seq.calculate_kspace()
    n = train.n_samples
    for e in range(4):
        seg = k_adc[0, e * n : (e + 1) * n]
        if e % 2 == 0:
            assert seg[0] < seg[-1]
        else:
            assert seg[0] > seg[-1]
        assert abs(seg[-1] - seg[0]) == pytest.approx(train._ro.k_width - train._ro.delta_k, rel=1e-6)


def test_flyback_echoes_all_same_polarity_and_retrace() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=3, flyback=True, spoil_position="none")
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    k_adc, *_ = seq.calculate_kspace()
    n = train.n_samples
    starts = [k_adc[0, e * n] for e in range(3)]
    ends = [k_adc[0, e * n + n - 1] for e in range(3)]
    assert np.ptp(starts) < 1e-6
    assert np.ptp(ends) < 1e-6
    assert train.esp > pp.calc_duration(train._ro.gx)  # flyback adds a gap beyond a bare echo


def test_multiecho_eco_labels_set_then_increment() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=3, spoil_position="none")
    seq = pp.Sequence(opts)
    train(seq, pe_idx=5)
    events = _labels(seq, "ECO")
    assert [(t, v) for _, t, v in events] == [("labelset", 0), ("labelinc", 1), ("labelinc", 1)]


def test_single_echo_has_no_eco_label() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=5)
    assert _labels(seq, "ECO") == []


def test_train_never_emits_lin_or_par_labels() -> None:
    opts = _opts()
    train = readout.Line3D(opts, (0.22, 0.22, 0.16), (64, 64, 16), num_echoes=2)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=3, par_idx=2)
    assert _labels(seq, "LIN") == []
    assert _labels(seq, "PAR") == []


def test_esp_s_pads_between_echoes() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, flyback=False)
    esp_min = train.esp
    padded = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=2, esp_s=esp_min + 2e-3)
    assert padded.esp == pytest.approx(esp_min + 2e-3, abs=1e-6)
    with pytest.raises(ValueError):
        readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=2, esp_s=1e-6)


# ----------------------------------------------------------------------
# 3D: PAR balance, RF-spoiling ADC phase, rf_phase_rad passthrough.
# ----------------------------------------------------------------------


def test_line3d_balances_pe_and_par_regardless_of_x_spoil() -> None:
    opts = _opts()
    train = readout.Line3D(
        opts, (0.22, 0.22, 0.16), (64, 64, 16), num_echoes=2, flyback=True, spoil_position="post"
    )
    seq = pp.Sequence(opts)
    train(seq, pe_idx=3, par_idx=2)
    _, k_full, *_ = seq.calculate_kspace()
    expected_x = train._ro.n_post * train._ro.delta_k + train._spoil_factor * train._ro.k_width
    assert k_full[0, -1] == pytest.approx(expected_x, rel=1e-6)
    assert k_full[1, -1] == pytest.approx(0.0, abs=1e-6)
    assert k_full[2, -1] == pytest.approx(0.0, abs=1e-6)


def test_rf_phase_rad_sets_every_echo_adc_phase() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=3)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=5, rf_phase_rad=1.2345)
    for i in range(1, seq.block_events.__len__() + 1):
        blk = seq.get_block(i)
        if blk.adc is not None:
            assert blk.adc.phase_offset == pytest.approx(1.2345)


# ----------------------------------------------------------------------
# Unbalanced spoiling must be a *bridged* trapezoid (gre.py's
# unbalanced_line construction), never a plain rewind-to-zero followed by a
# separate 0->spoil->0 lobe.
# ----------------------------------------------------------------------


def _gx_shapes(seq):
    """('trap'|'grad'|None, first, last) per block's x gradient (endpoints via calc_duration/waveform)."""
    out = []
    for i in range(1, seq.block_events.__len__() + 1):
        gx = seq.get_block(i).gx
        if not gx:
            out.append((None, None, None))
        elif gx.type == "trap":
            out.append(("trap", 0.0, 0.0))
        else:
            out.append(("grad", getattr(gx, "first", None), getattr(gx, "last", None)))
    return out


def test_balanced_uses_plain_trapezoids_both_sides() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="none")
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    shapes = _gx_shapes(seq)
    assert [s[0] for s in shapes] == ["trap", "trap", "trap"]


def test_post_spoil_bridges_last_echo_directly_into_spoiler() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="post", spoil_factor=1.0)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    types, firsts, lasts = zip(*_gx_shapes(seq))
    assert types == ("trap", "grad", "grad")  # plain prewind, split echo, bridge
    # echo's own block ends *at* plateau amplitude (no fall ramp) ...
    assert firsts[1] == pytest.approx(0.0)
    assert lasts[1] == pytest.approx(train._ro.gx.amplitude)
    # ... and the bridge picks up exactly there, no return through zero.
    assert firsts[2] == pytest.approx(lasts[1])
    assert lasts[2] == pytest.approx(0.0)


def test_pre_spoil_bridges_prewind_directly_into_first_echo() -> None:
    opts = _opts()
    train = readout.Line2D(opts, (0.22, 0.22), (64, 64), num_echoes=1, spoil_position="pre", spoil_factor=1.0)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    types, firsts, lasts = zip(*_gx_shapes(seq))
    assert types == ("grad", "grad", "trap")  # bridge, split echo, plain rewind
    assert firsts[0] == pytest.approx(0.0)
    assert lasts[0] == pytest.approx(train._ro.gx.amplitude)
    assert firsts[1] == pytest.approx(lasts[0])  # echo resumes exactly at the plateau
    assert lasts[1] == pytest.approx(0.0)


def test_monopolar_three_echo_pf1_matches_bridged_read_spoil_layout() -> None:
    """(pre, read, flbck, read, flbck, bridged_read_spoil), per the user's own example."""
    opts = _opts()
    train = readout.Line2D(
        opts, (0.22, 0.22), (64, 64), num_echoes=3, flyback=True, pf=1.0, spoil_position="post", spoil_factor=1.0
    )
    seq = pp.Sequence(opts)
    train(seq, pe_idx=10)
    types = [s[0] for s in _gx_shapes(seq)]
    assert types == ["trap", "trap", "trap", "trap", "trap", "grad", "grad"]
    # blocks 6+7 are the bridged (read, spoil) pair: split last echo -> spoiler,
    # continuous through the shared plateau amplitude, never back through zero.
    _, firsts, lasts = zip(*_gx_shapes(seq))
    assert lasts[5] == pytest.approx(firsts[6])
    assert lasts[6] == pytest.approx(0.0)
    ok, err = seq.check_timing()
    assert ok, err


def test_pre_spoil_bridge_right_aligns_when_pe_needs_more_time() -> None:
    """A tiny spoil (short bridge) next to a large PE matrix (long worst-case PE
    lobe) must delay the bridge's *start*, not pad after it -- it has to still
    land exactly at gx.amplitude when echo 0's split remainder takes over."""
    opts = _opts()
    train = readout.Line2D(opts, (0.2, 0.5), (64, 512), num_echoes=1, spoil_position="pre", spoil_factor=0.001)
    seq = pp.Sequence(opts)
    train(seq, pe_idx=0)  # edge of k-space -> worst-case (longest) PE lobe
    ok, err = seq.check_timing()
    assert ok, err
    b1, b2 = seq.get_block(1), seq.get_block(2)
    assert b1.gx.delay > 0.0  # bridge pushed later, not left at block start
    assert pp.calc_duration(b1.gx) == pytest.approx(train.t_prephase_s)
    assert b1.gx.last == pytest.approx(b2.gx.first)  # still continuous into echo 0
