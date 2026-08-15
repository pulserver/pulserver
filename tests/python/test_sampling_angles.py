"""The two angular conventions the rotated-readout plugins share.

Radial lines and PROPELLER blades are diametric, so they spread over a half
turn; spiral arms are one-sided, so they spread over the whole turn. These pin
that both the golden and the uniform builtins express each convention, and that
the full-turn golden angle is the classic 137.51 degrees the spiral plugins
used to hard-code.
"""

from __future__ import annotations

import numpy as np

from pulserver.pypulseq import calc_golden_angles, calc_uniform_angles


def test_default_golden_is_the_pi_periodic_radial_angle():
    step = np.diff(calc_golden_angles(2))[0]
    assert np.isclose(step, np.pi / ((1 + np.sqrt(5)) / 2))  # pi / phi, 111.25 deg


def test_full_circle_golden_is_the_classic_137_degree_angle():
    step = calc_golden_angles(2, full_circle=True)[1]
    assert np.isclose(np.rad2deg(step), 137.50776, atol=1e-4)
    # Exactly the pi*(3 - sqrt(5)) the spiral plugins used to hard-code.
    assert np.isclose(step, np.pi * (3.0 - np.sqrt(5.0)))


def test_full_circle_golden_matches_the_unwrapped_arm_sweep_as_rotations():
    golden = np.pi * (3.0 - np.sqrt(5.0))
    builtin = calc_golden_angles(21, full_circle=True)
    assert np.allclose(builtin, (np.arange(21) * golden) % (2 * np.pi))


def test_uniform_span_selects_half_or_whole_turn():
    assert np.allclose(calc_uniform_angles(6), np.arange(6) * 2 * np.pi / 6)
    assert np.allclose(calc_uniform_angles(6, span=np.pi), np.arange(6) * np.pi / 6)


def test_diametric_uniform_never_repeats_a_direction():
    # A half-turn span keeps every spoke a distinct line (mod pi).
    angles = calc_uniform_angles(8, span=np.pi)
    assert len(np.unique(np.round(angles % np.pi, 9))) == 8
