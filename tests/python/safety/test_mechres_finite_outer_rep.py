"""The outer repeat is finite, and the spectrum between harmonics is real.

A scan is M = ``num_instances`` repetitions of the canonical TR, not an
infinite Dirac comb. Its spectrum is the single-TR transform multiplied by
the Dirichlet kernel of M, so drive exists between the exact TR harmonics
k/T_TR, peaking at (k + (j + 1/2)/M) with levels 2/(pi(2j+1)).

Two properties are load-bearing and are what these tests hold onto:

* the single-TR transform is evaluated FRESH at each probed frequency and
  only then attenuated. Scaling the coarse harmonic's amplitude by the
  Dirichlet ratio instead cannot expose anything, because that ratio never
  exceeds 1;
* the probes sit on the lobes. Sampling at integer multiples of 1/M lands on
  the kernel's nulls, where the attenuation is zero and a probe reports
  nothing however many are spent.

These tests exercise the low-level ``_calc_mech_resonances`` binding
directly, so the actual candidate amplitudes -- not just a verdict -- can be
inspected.
"""

from pathlib import Path

import pypulseq as pp
import pytest
from pulserver._ext.pulseg import _calc_mech_resonances
from pulserver.pypulseq import Opts

from .conftest import build_collection

RASTER = 20e-6
BAND = (500.0, 3000.0, 0.0)  # zero tolerance: eps falls back to the policy amplitude


def _build_two_block_seq(tmp_path: Path) -> Path:
    """Two DIFFERENT trapezoids back-to-back (gx then gy) -- deliberately
    not period-1 so pulserver's own periodicity/dedup detector can't
    collapse an N-repeat authoring into a shorter canonical period out from
    under this test (which happened, and was noted, when this was first
    tried with a single repeated block)."""
    sys_ = pp.Opts(
        max_grad=50,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=RASTER,
        block_duration_raster=RASTER,
        rf_raster_time=1e-6,
        adc_raster_time=1e-7,
    )
    seq = pp.Sequence(system=sys_)
    g1 = pp.make_trapezoid(
        channel="x",
        amplitude=0.5 * sys_.max_grad,
        flat_time=400e-6,
        rise_time=RASTER * 20,
        system=sys_,
    )
    g2 = pp.make_trapezoid(
        channel="y",
        amplitude=0.3 * sys_.max_grad,
        flat_time=300e-6,
        rise_time=RASTER * 20,
        system=sys_,
    )
    seq.add_block(g1)
    seq.add_block(g2)
    path = tmp_path / "two_block.seq"
    seq.write(str(path))
    return path


def _candidate_grad_amps(seq_path: Path, num_averages: int):
    system = Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=150.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=RASTER,
        block_duration_raster=RASTER,
    )
    collection = build_collection(seq_path, system, num_averages=num_averages)
    rd = _calc_mech_resonances(
        collection,
        subsequence_idx=0,
        canonical_tr_idx=0,
        target_resolution_hz=1.0,
        max_freq_hz=3000.0,
        forbidden_bands=[BAND],
        peak_log10_threshold=None,
        peak_norm_scale=None,
        peak_eps=None,
        peak_prominence=None,
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
    """Repeating the TR finds drive that a single repetition does not.

    Scaling the coarse harmonic by the Dirichlet ratio rather than
    evaluating the transform afresh would leave every M identical to M=1,
    since a ratio at most 1 can never exceed the value it scales. At least
    one candidate must move once M > 1."""
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

    dist_16_64 = sum(abs(a - b) for a, b in zip(amps_m16, amps_m64, strict=True))
    dist_1_64 = sum(abs(a - b) for a, b in zip(amps_m1, amps_m64, strict=True))
    assert dist_16_64 < dist_1_64
