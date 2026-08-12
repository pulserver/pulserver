"""``sim_rf``: a Bloch simulation of one RF pulse, across off-resonance.

A port of MATLAB's ``mr.simRf``, which upstream PyPulseq has no equivalent of,
so there is no second implementation to diff against. What is checked instead
is physics the pulses themselves state: a 90 tips fully and a 180 inverts, a
refocusing pulse refocuses where an excitation pulse does not, and the width
of the profile is the bandwidth the pulse was designed for.
"""

from __future__ import annotations

import numpy as np
import pulserver.pypulseq as pp
import pytest


@pytest.fixture
def system():
    return pp.Opts(rf_raster_time=1e-6)


def _pulse(system, flip_deg, use, *, time_bw_product=4, duration=2e-3):
    return pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration,
        time_bw_product=time_bw_product,
        system=system,
        use=use,
        return_gz=False,
    )


def _on_resonance(response):
    return int(np.argmin(np.abs(response.frequency)))


def test_a_ninety_tips_fully_and_a_one_eighty_inverts(system):
    excitation = pp.sim_rf(_pulse(system, 90, "excitation"), compat=False)
    at = _on_resonance(excitation)
    assert excitation.mz_z[at] == pytest.approx(0.0, abs=1e-3)
    assert abs(excitation.mz_xy[at]) == pytest.approx(1.0, abs=1e-3)

    inversion = pp.sim_rf(_pulse(system, 180, "inversion"), compat=False)
    at = _on_resonance(inversion)
    assert inversion.mz_z[at] == pytest.approx(-1.0, abs=1e-3)
    assert abs(inversion.mz_xy[at]) == pytest.approx(0.0, abs=1e-3)


def test_refocusing_efficiency_separates_a_180_from_a_90(system):
    """``ref_eff`` is the number ``Mz`` cannot give you.

    A 90 and a 180 both move Mz, but only the 180 refocuses transverse
    magnetisation -- which is the whole question for a spin-echo train, and
    the reason mx_xy and my_xy are simulated at all.
    """
    refocusing = pp.sim_rf(_pulse(system, 180, "refocusing"), compat=False)
    excitation = pp.sim_rf(_pulse(system, 90, "excitation"), compat=False)

    assert abs(refocusing.ref_eff[_on_resonance(refocusing)]) == pytest.approx(1.0, abs=1e-3)
    assert abs(excitation.ref_eff[_on_resonance(excitation)]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize(
    ("time_bw_product", "duration"),
    [(4, 2e-3), (8, 4e-3), (4, 1e-3), (6, 3e-3)],
)
def test_the_simulated_profile_is_as_wide_as_the_pulse_was_designed_to_be(
    system, time_bw_product, duration
):
    """FWHM of the excitation profile against the nominal ``TBW / duration``.

    This is the check that the integration is right and not merely
    self-consistent: the answer has to land on a number chosen before the
    simulator ran.
    """
    response = pp.sim_rf(_pulse(system, 90, "excitation", time_bw_product=time_bw_product,
                                duration=duration), compat=False)
    magnitude = np.abs(response.mz_xy)
    above = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
    measured = response.frequency[above[-1]] - response.frequency[above[0]]

    assert measured == pytest.approx(time_bw_product / duration, rel=0.03)


def test_a_frequency_offset_moves_the_profile_and_not_its_width(system):
    offset = 1500.0
    centred = pp.sim_rf(_pulse(system, 90, "excitation"), compat=False)
    shifted_pulse = _pulse(system, 90, "excitation")
    shifted_pulse.freq_offset = offset
    shifted = pp.sim_rf(shifted_pulse, compat=False)

    def centre_and_width(response):
        magnitude = np.abs(response.mz_xy)
        above = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
        edges = response.frequency[above[[0, -1]]]
        return 0.5 * (edges[0] + edges[1]), edges[1] - edges[0]

    centre_a, width_a = centre_and_width(centred)
    centre_b, width_b = centre_and_width(shifted)
    assert centre_b - centre_a == pytest.approx(offset, rel=0.05)
    assert width_b == pytest.approx(width_a, rel=0.05)


def test_compat_returns_matlabs_six_tuple_in_matlabs_order(system):
    rf = _pulse(system, 90, "excitation")
    mz_z, mz_xy, frequency, ref_eff, mx_xy, my_xy = pp.sim_rf(rf)
    typed = pp.sim_rf(rf, compat=False)

    np.testing.assert_array_equal(mz_z, typed.mz_z)
    np.testing.assert_array_equal(mz_xy, typed.mz_xy)
    np.testing.assert_array_equal(frequency, typed.frequency)
    np.testing.assert_array_equal(ref_eff, typed.ref_eff)
    np.testing.assert_array_equal(mx_xy, typed.mx_xy)
    np.testing.assert_array_equal(my_xy, typed.my_xy)

    assert not hasattr(typed, "__iter__")
    assert not hasattr(typed, "__getitem__")


def test_the_rephase_factor_defaults_by_the_pulses_use(system):
    """A refocusing pulse is not rephased; an excitation is.

    Left unrephased, a slice-selective excitation carries the linear phase
    ramp the pulse leaves across the profile, so the transverse
    magnetisation cancels when it is summed -- which is exactly what the
    default is there to avoid.
    """
    rf = _pulse(system, 90, "excitation")
    rephased = np.abs(pp.sim_rf(rf, compat=False).mz_xy.sum())
    unrephased = np.abs(pp.sim_rf(rf, 0.0, compat=False).mz_xy.sum())
    assert rephased > 3.0 * unrephased
