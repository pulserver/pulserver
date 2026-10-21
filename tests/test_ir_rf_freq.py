"""The RF frequency offsets a position within the repetition plays across repetitions: whether they vary, and their mean."""

import math

import numpy as np
import pypulseqpp as pp

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

SYSTEM = pp.Opts(B0=3.0)
THICKNESS = 5e-3


def _stack(tmp_path, offsets_hz, fov_offset=None, name="stack.seq"):
    """One selective excitation and a rest per slice, the slice at each carrier offset: a repetition per slice."""
    seq = pp.Sequence(SYSTEM)
    for offset in offsets_hz:
        pulse, gz, _ = pp.make_sinc_pulse(
            math.pi / 8,
            duration=2e-3,
            slice_thickness=THICKNESS,
            freq_offset=offset,
            return_gz=True,
            use="excitation",
            system=SYSTEM,
        )
        seq.add_block(pulse, gz)
        seq.add_block(pp.make_delay(5e-3))
    path = tmp_path / name
    seq.write(str(path))
    ir.convert(path, SYSTEM, fov_offset=fov_offset)
    return path, gz.amplitude


def _excitations(played):
    return played["rf_amp_hz"] != 0


def test_a_stack_is_flagged_and_reports_the_mean_of_its_offsets(tmp_path):
    offsets = [-1000.0, 0.0, 2500.0, 500.0]
    path, _ = _stack(tmp_path, offsets)
    played = ir.play(path)
    rf = _excitations(played)

    assert np.unique(played["segment"][rf]).size == 1
    np.testing.assert_array_equal(played["rf_freq_hz"][rf], offsets)
    assert played["rf_freq_variable"][rf].tolist() == [1] * len(offsets)
    assert played["rf_freq_mean_hz"][rf].tolist() == [500.0] * len(offsets)
    assert played["rf_freq_variable"][~rf].tolist() == [0] * len(offsets)
    assert played["rf_freq_mean_hz"][~rf].tolist() == [0.0] * len(offsets)


def test_a_single_slice_off_centre_is_not_flagged(tmp_path):
    """Every instance plays one offset: there is nothing to average away."""
    path, _ = _stack(tmp_path, [1234.5] * 3)
    played = ir.play(path)
    rf = _excitations(played)

    assert played["rf_freq_variable"].tolist() == [0] * 6
    assert played["rf_freq_mean_hz"][rf].tolist() == [1234.5] * 3


def test_offsets_within_half_a_hertz_are_one_offset(tmp_path):
    path, _ = _stack(tmp_path, [100.0, 100.25, 100.5])
    played = ir.play(path)

    assert played["rf_freq_variable"].tolist() == [0] * 6


def test_a_shifted_stack_reports_the_mean_of_its_shifted_offsets(tmp_path):
    """The offsets are those the cache plays, the field-of-view offset included."""
    spacing = 2 * THICKNESS
    shift = 0.02
    centred, amplitude = _stack(tmp_path, [0.0] * 5, name="centred.seq")
    designed = amplitude * spacing * np.arange(-2, 3)
    path, _ = _stack(tmp_path, designed, fov_offset=(0.0, 0.0, shift))
    played = ir.play(path)
    rf = _excitations(played)
    offsets = played["rf_freq_hz"][rf]

    np.testing.assert_allclose(offsets, designed + amplitude * shift, rtol=1e-6)
    assert played["rf_freq_variable"][rf].all()
    np.testing.assert_allclose(
        played["rf_freq_mean_hz"][rf], np.mean(offsets.astype(np.float64)), rtol=1e-6
    )
    np.testing.assert_allclose(
        played["rf_freq_mean_hz"][rf], amplitude * shift, rtol=1e-5
    )
    assert (ir.play(centred)["rf_freq_mean_hz"] == 0).all()


def test_the_prescan_plays_a_varying_offset_at_its_mean(tmp_path):
    offsets = [-1000.0, 0.0, 2500.0, 500.0]
    path, _ = _stack(tmp_path, offsets)
    budget = WaveBudget(10**6, 4.0)
    prescan = ir.playout(path, budget, prescan=Prescan(0, 10**6))["blocks"]
    scan = ir.playout(path, budget)["blocks"]
    rf = prescan["rf_amp_hz"] != 0

    np.testing.assert_array_equal(scan["rf_freq_hz"][scan["rf_amp_hz"] != 0], offsets)
    assert (prescan["rf_freq_variable"][rf] == 1).all()
    assert (prescan["rf_freq_hz"][rf] == 500.0).all()
