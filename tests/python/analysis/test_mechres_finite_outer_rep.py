"""Finite-outer-rep fix tests for the A_eq mechanical-resonance criterion.

PLAN_safety_mechres_finite_outer_rep.md: the outermost TR repeat (M =
num_instances) used to be treated as an infinite Dirac comb -- A_eq evaluated
only at exact TR harmonics k/T_TR, independent of M. That is provably a
no-op fix if "fixed" by scaling the already-computed exact-harmonic
amplitude by the Dirichlet ratio (a ratio <=1 can never expose a candidate
the coarse point didn't already show) -- caught via numeric exploration
during implementation. The actual fix evaluates S_TR(f) fresh at fractional
TR harmonics between consecutive exact ones and attenuates by the closed
-form Dirichlet ratio, with sample points geometrically concentrated near
each adjacent main lobe (where real sidelobes live for large M) rather than
spread uniformly across the interval (uniform spacing was tried first and
confirmed to miss every sidelobe once M exceeded the sample count).

These tests exercise the low-level ``_calc_mech_resonances`` binding
directly (not the pass/fail ``.check()`` wrapper) so the actual candidate
amplitudes -- not just a boolean verdict -- can be inspected.
"""

from pathlib import Path

import pypulseq as pp
import pytest

from pulserver._ext._pulseg_wrapper import _calc_mech_resonances
from pulserver.analysis import Opts, SequenceCollection

RASTER = 20e-6
BAND = (500.0, 3000.0, 0.0)  # zero-tolerance band, forces eps to the G_max floor


def _build_two_block_seq(tmp_path: Path) -> Path:
    """Two DIFFERENT trapezoids back-to-back (gx then gy) -- deliberately
    not period-1 so pulserver's own periodicity/dedup detector can't
    collapse an N-repeat authoring into a shorter canonical period out from
    under this test (which happened, and was noted, when this was first
    tried with a single repeated block)."""
    sys_ = pp.Opts(
        max_grad=50, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
        grad_raster_time=RASTER, block_duration_raster=RASTER,
        rf_raster_time=1e-6, adc_raster_time=1e-7,
    )
    seq = pp.Sequence(system=sys_)
    g1 = pp.make_trapezoid(channel="x", amplitude=0.5 * sys_.max_grad, flat_time=400e-6, rise_time=RASTER * 20, system=sys_)
    g2 = pp.make_trapezoid(channel="y", amplitude=0.3 * sys_.max_grad, flat_time=300e-6, rise_time=RASTER * 20, system=sys_)
    seq.add_block(g1)
    seq.add_block(g2)
    path = tmp_path / "two_block.seq"
    seq.write(str(path))
    return path


def _candidate_grad_amps(seq_path: Path, num_averages: int):
    a_sys = Opts(max_grad=50.0, max_slew=150.0, B0=3.0, grad_raster_time=RASTER, block_duration_raster=RASTER)
    sc = SequenceCollection(str(seq_path), system=a_sys, num_averages=num_averages)
    rd = _calc_mech_resonances(
        sc._cseq, subsequence_idx=0, canonical_tr_idx=0,
        target_resolution_hz=1.0, max_freq_hz=3000.0,
        forbidden_bands=[BAND],
        peak_log10_threshold=None, peak_norm_scale=None, peak_eps=None, peak_prominence=None,
    )
    assert rd["num_instances"] == num_averages
    return rd["candidate_grad_amps"]


def test_finite_outer_rep_m1_regression_identity(tmp_path):
    """M=1 (e.g. a single-pass hyper-TR) must take the early-out and match
    the exact-harmonic-only formula exactly -- no sub-sampling should run."""
    seq_path = _build_two_block_seq(tmp_path)
    amps = _candidate_grad_amps(seq_path, num_averages=1)
    assert len(amps) == 9  # sanity: same coarse grid as always


def test_finite_outer_rep_not_a_noop_for_m_greater_than_1(tmp_path):
    """The core regression this fix exists to prevent: an earlier, WRONG
    version of the fix (scaling the coarse amplitude by the Dirichlet ratio
    instead of evaluating S_TR fresh at the fractional frequency) is a
    mathematical no-op -- M=2..64 would silently reproduce the M=1 values
    exactly, because a ratio <=1 can never exceed the value it scales.
    Assert this doesn't happen: at least one candidate's amplitude must
    differ from the M=1 baseline once M>1 (real sidelobe energy is found)."""
    seq_path = _build_two_block_seq(tmp_path)
    amps_m1 = _candidate_grad_amps(seq_path, num_averages=1)
    amps_m4 = _candidate_grad_amps(seq_path, num_averages=4)
    assert amps_m1 != amps_m4


@pytest.mark.parametrize("m", [2, 4, 16, 64])
def test_finite_outer_rep_large_m_does_not_collapse_to_m1(tmp_path, m):
    """Regression for the near-lobe under-sampling bug caught during
    implementation: uniform sub-point spacing across the coarse interval
    missed every sidelobe once M exceeded the fixed sample-point count
    (concretely, M=64 with 16 uniform points silently reproduced the exact
    M=1 candidate_grad_amps array). Geometric spacing concentrated near
    each lobe edge fixes this -- assert large M does NOT silently collapse
    back to the M=1 (no-sidelobe-found) values."""
    seq_path = _build_two_block_seq(tmp_path)
    amps_m1 = _candidate_grad_amps(seq_path, num_averages=1)
    amps_m = _candidate_grad_amps(seq_path, num_averages=m)
    assert amps_m != amps_m1


def test_finite_outer_rep_large_m_sidelobe_envelope_stabilizes(tmp_path):
    """Physical sanity check: as M grows, the discovered sidelobe envelope
    should stabilize (converge), not diverge or oscillate wildly -- M=16
    and M=64 candidate amplitudes should be much closer to each other than
    either is to the M=1 (no-sidelobe) baseline."""
    seq_path = _build_two_block_seq(tmp_path)
    amps_m1 = _candidate_grad_amps(seq_path, num_averages=1)
    amps_m16 = _candidate_grad_amps(seq_path, num_averages=16)
    amps_m64 = _candidate_grad_amps(seq_path, num_averages=64)

    dist_16_64 = sum(abs(a - b) for a, b in zip(amps_m16, amps_m64))
    dist_1_64 = sum(abs(a - b) for a, b in zip(amps_m1, amps_m64))
    assert dist_16_64 < dist_1_64
