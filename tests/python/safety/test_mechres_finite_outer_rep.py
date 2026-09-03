r"""The verdict is a window reading, and a window does not count repetitions.

A mode integrates the drive over its own memory, $1/\Delta f$, so what the
check reads at a frequency is the amplitude sustained over that window --
computed from the canonical TR laid out over as many repetitions as one
window can reach, however many the scan really plays. Between the TR
harmonics the window's own finite length leaves real drive, which the fine
grid across the band reads.

These tests exercise the low-level ``_calc_mech_resonances`` binding
directly, so the candidate amplitudes -- not just a verdict -- can be
inspected.
"""

from itertools import pairwise
from pathlib import Path

import pypulseq as pp
from pulserver._ext.pulseg import _calc_mech_resonances, _mech_scan_window_probe
from pulserver.pypulseq import Opts

from .conftest import build_collection

RASTER = 20e-6
BAND = (500.0, 3000.0, 0.0)  # zero tolerance: eps falls back to the policy amplitude


def _build_two_block_seq(tmp_path: Path, repeats: int = 1) -> Path:
    """``repeats`` TRs of two DIFFERENT trapezoids back-to-back (gx then gy).

    Two different blocks, deliberately: a period-1 authoring lets pulserver's
    own periodicity/dedup detector collapse the repeats into a shorter
    canonical period, and then M is not what the test asked for.
    """
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
    for _ in range(repeats):
        seq.add_block(g1)
        seq.add_block(g2)
    path = tmp_path / f"two_block_x{repeats}.seq"
    seq.write(str(path))
    return path


def _candidates(seq_path: Path, expected_m: int):
    system = Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=150.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=RASTER,
        block_duration_raster=RASTER,
    )
    collection = build_collection(seq_path, system)
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
    assert rd["num_instances"] == expected_m
    return rd


def _amps(seq_path: Path, expected_m: int):
    rd = _candidates(seq_path, expected_m)
    return list(rd["candidate_grad_amps"])


def test_repetitions_add_to_the_reading_until_the_memory_is_full(tmp_path):
    """A 1.6 ms repetition under a 20 ms memory: one repetition reads its own
    transform over the memory, four read four times more, and once the window
    holds every copy it can (sixteen and beyond) the reading stops changing."""
    amps_m1 = _amps(_build_two_block_seq(tmp_path, 1), expected_m=1)
    amps_m4 = _amps(_build_two_block_seq(tmp_path, 4), expected_m=4)
    amps_m16 = _amps(_build_two_block_seq(tmp_path, 16), expected_m=16)
    amps_m64 = _amps(_build_two_block_seq(tmp_path, 64), expected_m=64)
    i = max(range(len(amps_m64)), key=lambda k: amps_m64[k])
    assert amps_m1[i] < amps_m4[i] < amps_m16[i]
    assert amps_m16 == amps_m64


def test_drive_between_the_harmonics_is_read(tmp_path):
    """The grid is finer than the TR comb, and the window leaves drive there."""
    rd = _candidates(_build_two_block_seq(tmp_path, 4), expected_m=4)
    freqs = list(rd["candidate_freqs"])
    amps = list(rd["candidate_grad_amps"])
    t_tr_s = rd["tr_duration_us"] * 1e-6 if "tr_duration_us" in rd else None
    assert len(freqs) > 100
    off_harmonic = [
        a
        for f, a in zip(freqs, amps, strict=True)
        if t_tr_s is None or abs(f * t_tr_s - round(f * t_tr_s)) > 0.1
    ]
    assert max(off_harmonic) > 0.0


def test_the_candidates_cover_the_guarded_band_at_the_fine_spacing(tmp_path):
    """A 2500 Hz band gets no guard (it is a keep-out range, not a mode) and
    a fixed point count; the candidates run from its lower to its upper edge."""
    rd = _candidates(_build_two_block_seq(tmp_path, 1), expected_m=1)
    freqs = list(rd["candidate_freqs"])
    assert abs(freqs[0] - BAND[0]) < 1e-3
    assert abs(freqs[-1] - BAND[1]) < 1e-3
    spacing = {round(b - a, 6) for a, b in pairwise(freqs)}
    assert len(spacing) == 1


def test_the_periodic_reading_is_the_whole_scans_reading(tmp_path):
    """With nothing varying, the TR tiled into the memory reads what the
    windows slid over every event of the real scan read, at the same 20 ms
    memory: the two regimes are one criterion, and the Bernstein factor on
    the grid is the whole difference."""
    seq_path = _build_two_block_seq(tmp_path, 8)
    rd = _candidates(seq_path, expected_m=8)
    freqs = list(rd["candidate_freqs"])
    system = Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=150.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=RASTER,
        block_duration_raster=RASTER,
    )
    collection = build_collection(seq_path, system)
    picks = [
        min(range(len(freqs)), key=lambda j: abs(freqs[j] - f))
        for f in range(1500, 2100, 100)
    ]
    grids = [(float(freqs[i]), 0.0, 1) for i in picks]
    probe = _mech_scan_window_probe(collection, grids, 20000.0, 0, 0)
    for k, i in enumerate(picks):
        for axis in "xyz":
            periodic = rd[f"candidate_amps_g{axis}"][i]
            scan = probe[f"amp_g{axis}"][k]
            assert abs(periodic - scan) <= 0.01 * max(scan, 1.0) + 1e-3, (
                axis,
                freqs[i],
                periodic,
                scan,
            )
